package oneocr.onnx;

import java.io.IOException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.function.IntConsumer;

import ai.onnxruntime.OrtException;

/**
 * The one thread pool this engine uses, shared by every stage and every {@link OneOcrOnnx}
 * instance in the process.
 *
 * <p>It is deliberately a single shared pool rather than one per engine. A caller that holds
 * several engines — or that reads several pages at once — would otherwise get a pool per engine
 * and oversubscribe the machine, which on this workload costs more than the parallelism gains.
 * Sizing is {@link Threads#lines()}; setting {@code oneocr.threads.lines=1} disables the pool
 * entirely and everything runs on the calling thread, which is what a caller with its own pool
 * over pages should do.
 *
 * <p>ForkJoin is the right shape here because the work nests: a caller may already be inside a
 * pool task, a page fans out over its lines, and a line fans out over the rows of an image
 * resize. A joining ForkJoin worker helps run other tasks instead of blocking, so nesting costs
 * nothing and cannot deadlock the pool. Its threads are daemons, so nothing has to be shut down.
 */
public final class Workers {
    static final int CHUNKS_PER_THREAD = 4;

    private static volatile ForkJoinPool shared;

    private Workers() {}

    static ForkJoinPool pool() {
        var existing = shared;
        if (existing != null) return existing;
        synchronized (Workers.class) {
            if (shared == null) shared = new ForkJoinPool(Threads.lines());
            return shared;
        }
    }

    /** Whether there is any point splitting work at all. */
    static boolean parallel() {
        return Threads.lines() > 1;
    }

    /**
     * Runs {@code body} for every index in {@code [0, count)}, in parallel when that is worth it.
     * Indices are disjoint, so a body that writes only to its own slice of an output array needs
     * no synchronisation and produces bit-identical results to the sequential loop.
     */
    static void split(int count, IntConsumer body) {
        if (count <= 0) return;
        if (count == 1 || !parallel()) {
            for (var i = 0; i < count; i++) body.accept(i);
            return;
        }
        var chunk = Math.max(1, count / (Threads.lines() * CHUNKS_PER_THREAD));
        pool().invoke(new Range(0, count, chunk, body));
    }

    /**
     * Runs {@code task} for every index in {@code [0, count)} and returns the results in index
     * order, so a parallel run produces the same ordering as a sequential one.
     */
    static <T> Object[] map(int count, Task<T> task) throws OrtException, IOException {
        var out = new Object[count];
        if (count == 0) return out;
        if (count == 1 || !parallel()) {
            for (var i = 0; i < count; i++) out[i] = task.run(i);
            return out;
        }
        try {
            split(count, i -> {
                try {
                    out[i] = task.run(i);
                } catch (OrtException | IOException checked) {
                    throw new Wrapped(checked);
                }
            });
        } catch (Wrapped wrapped) {
            if (wrapped.getCause() instanceof OrtException ort) throw ort;
            throw (IOException) wrapped.getCause();
        }
        return out;
    }

    interface Task<T> {
        T run(int index) throws OrtException, IOException;
    }

    /** Carries a checked exception out of a pool task so {@link #map} can rethrow it as declared. */
    static final class Wrapped extends RuntimeException {
        Wrapped(Exception cause) {
            super(cause);
        }
    }

    static final class Range extends RecursiveAction {
        final int from, to, chunk;
        final IntConsumer body;

        Range(int from, int to, int chunk, IntConsumer body) {
            this.from = from;
            this.to = to;
            this.chunk = chunk;
            this.body = body;
        }

        @Override
        protected void compute() {
            if (to - from <= chunk) {
                for (var i = from; i < to; i++) body.accept(i);
                return;
            }
            var middle = (from + to) >>> 1;
            invokeAll(new Range(from, middle, chunk, body), new Range(middle, to, chunk, body));
        }
    }
}
