package oneocr.onnx;

import javax.imageio.ImageIO;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Hammers ONE {@link OneOcrOnnx} from many threads at once and checks every result against the
 * single-threaded answer for the same page.
 *
 * <p>This is the test behind the claim that {@code recognize} is now safe to call concurrently — the
 * thing PRP 27's consumer worked around with a single worker thread, and what lets a caller read
 * several pages at once. It deliberately mixes page sizes so that pages of different cost overlap,
 * and starts from a cold engine on the first pass so the lazily created recognizer sessions are
 * raced rather than pre-warmed.
 */
public class ConcurrencyProbe {
    public static void main(String[] args) throws Exception {
        var threads = Integer.getInteger("probe.threads", 8);
        var passes = Integer.getInteger("probe.passes", 4);

        var images = new ArrayList<Raster>();
        var names = new ArrayList<String>();
        for (var file : args) {
            images.add(Raster.of(ImageIO.read(Path.of(file).toFile())));
            names.add(Path.of(file).getFileName().toString());
        }
        if (images.isEmpty()) throw new IllegalArgumentException("give me some images");

        System.out.printf("%d threads, %d passes over %d images, one shared engine%n",
                threads, passes, images.size());

        List<String> expected;
        try (var engine = new OneOcrOnnx(ModelPaths.resolve(null))) {
            System.setProperty(Threads.LINES, "1");
            expected = new ArrayList<>();
            for (var image : images) expected.add(engine.recognize(image, null, 0).fullText());
            System.clearProperty(Threads.LINES);
            for (var i = 0; i < images.size(); i++)
                System.out.printf("  reference  %-18s %,d chars%n", names.get(i), expected.get(i).length());
        }

        var mismatches = new AtomicInteger();
        var done = new AtomicInteger();
        // A fresh engine, so the first concurrent calls race session creation rather than finding it done.
        try (var engine = new OneOcrOnnx(ModelPaths.resolve(null));
             var pool = Executors.newFixedThreadPool(threads)) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>();
            for (var pass = 0; pass < passes; pass++) {
                for (var i = 0; i < images.size(); i++) {
                    var index = i;
                    futures.add(pool.submit(() -> {
                        var text = engine.recognize(images.get(index), null, 0).fullText();
                        if (!text.equals(expected.get(index))) {
                            mismatches.incrementAndGet();
                            System.out.printf("  MISMATCH   %s: %,d chars, expected %,d%n",
                                    names.get(index), text.length(), expected.get(index).length());
                        }
                        done.incrementAndGet();
                        return null;
                    }));
                }
            }
            for (var f : futures) f.get();
        }
        System.out.printf("%s  %d calls, %d mismatches%n",
                mismatches.get() == 0 ? "PASS" : "FAIL", done.get(), mismatches.get());
        System.exit(mismatches.get() == 0 ? 0 : 1);
    }
}
