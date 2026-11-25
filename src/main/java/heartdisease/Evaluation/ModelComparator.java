
package heartdisease.Evaluation;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import heartdisease.ModelImprovement.ImprovementResult;

public final class ModelComparator {

    private ModelComparator() {
        // Utility class
    }

    // Store a baseline/improved comparison
    public static class Comparison {

        private final ImprovementResult baseline;
        private final ImprovementResult improved;

        private Comparison(
                ImprovementResult baseline,
                ImprovementResult improved) {

            this.baseline = baseline;
            this.improved = improved;
        }

        public String toCsvRow() {
            List<String> cells = new ArrayList<>();

            cells.add(escape(baseline.getModelName()));
            cells.add(escape(improved.getMethodName()));

            addMetric(cells,
                    baseline.getAccuracy(),
                    improved.getAccuracy());

            addMetric(cells,
                    baseline.getPrecision(),
                    improved.getPrecision());

            addMetric(cells,
                    baseline.getRecall(),
                    improved.getRecall());

            addMetric(cells,
                    baseline.getF1Score(),
                    improved.getF1Score());

            addMetric(cells,
                    baseline.getRocAuc(),
                    improved.getRocAuc());

            return String.join(",", cells);
        }
    }

    // ==========================================
    // COMPARE TWO RESULTS
    // ==========================================

    public static Comparison compare(
            ImprovementResult baseline,
            ImprovementResult improved) {

        if (baseline == null || improved == null) {
            throw new IllegalArgumentException(
                    "Evaluation results cannot be null."
            );
        }

        if (!baseline.getModelName().equals(
                improved.getModelName())) {

            throw new IllegalArgumentException(
                    "Models must have the same name."
            );
        }

        System.out.println("\n================================");
        System.out.println("MODEL: " + baseline.getModelName());
        System.out.println("IMPROVEMENT: " + improved.getMethodName());
        System.out.println("================================");

        System.out.printf(
                "%-18s %12s %12s %12s%n",
                "Metric", "Baseline", "Improved", "Delta"
        );

        printMetric("Accuracy (%)",
                baseline.getAccuracy(),
                improved.getAccuracy());

        printMetric("Precision",
                baseline.getPrecision(),
                improved.getPrecision());

        printMetric("Recall",
                baseline.getRecall(),
                improved.getRecall());

        printMetric("F1-score",
                baseline.getF1Score(),
                improved.getF1Score());

        printMetric("ROC-AUC",
                baseline.getRocAuc(),
                improved.getRocAuc());

        return new Comparison(baseline, improved);
    }

    private static void printMetric(
            String name, double before, double after) {

        System.out.printf(
                "%-18s %12s %12s %12s%n",
                name,
                format(before),
                format(after),
                formatDelta(before, after)
        );
    }

    // ==========================================
    // CSV EXPORT
    // ==========================================

    public static void saveCsv(
            List<Comparison> comparisons,
            String filePath) throws IOException {

        if (comparisons == null || comparisons.isEmpty()) {
            throw new IllegalArgumentException(
                    "Comparison list is empty."
            );
        }

        String header =
                "Model,Improvement,"
                + "AccuracyBaseline,AccuracyImproved,DeltaAccuracyPP,"
                + "PrecisionBaseline,PrecisionImproved,DeltaPrecision,"
                + "RecallBaseline,RecallImproved,DeltaRecall,"
                + "F1Baseline,F1Improved,DeltaF1,"
                + "RocAucBaseline,RocAucImproved,DeltaRocAuc";

        List<String> rows = new ArrayList<>();
        rows.add(header);

        for (Comparison comparison : comparisons) {
            rows.add(comparison.toCsvRow());
        }

        Path path = Path.of(filePath);
        Path parent = path.getParent();

        if (parent != null) {
            Files.createDirectories(parent);
        }

        Files.write(
                path,
                rows,
                StandardCharsets.UTF_8
        );

        System.out.println(
                "\nComparison CSV saved to: "
                + path.toAbsolutePath()
        );
    }

    private static void addMetric(
            List<String> cells,
            double before,
            double after) {

        cells.add(format(before));
        cells.add(format(after));

        if (Double.isFinite(before)
                && Double.isFinite(after)) {

            cells.add(format(after - before));
        } else {
            cells.add("N/A");
        }
    }

    private static String format(double value) {
        if (!Double.isFinite(value)) {
            return "N/A";
        }

        return String.format(Locale.US, "%.4f", value);
    }

    private static String formatDelta(
            double before, double after) {

        if (!Double.isFinite(before)
                || !Double.isFinite(after)) {
            return "N/A";
        }

        return String.format(
                Locale.US,
                "%+.4f",
                after - before
        );
    }

    private static String escape(String text) {
        if (text == null) {
            return "\"\"";
        }

        return "\""
                + text.replace("\"", "\"\"")
                + "\"";
    }
}
