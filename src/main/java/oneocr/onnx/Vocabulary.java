package oneocr.onnx;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;

public final class Vocabulary {
    final String[] entries;

    Vocabulary(String[] entries) {
        this.entries = entries;
    }

    public int size() {
        return entries.length;
    }

    public String charAt(int index) {
        return index >= 0 && index < entries.length && entries[index] != null ? entries[index] : "";
    }

    public static Vocabulary load(Path file, int vocabSize) throws IOException {
        var entries = new String[vocabSize];
        for (var line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (line.isEmpty()) continue;
            var colon = line.indexOf(':');
            if (colon <= 0) continue;
            var index = parse(line.substring(0, colon));
            if (index < 0 || index >= vocabSize) continue;
            var value = line.substring(colon + 1);
            entries[index] = value.startsWith(" ") ? value.substring(1) : value;
        }
        return new Vocabulary(entries);
    }

    static int parse(String s) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }
}
