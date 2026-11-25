
package heartdisease.Evaluation;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import heartdisease.ModelImprovement.ImprovementResult;

public class ModelComparatorTest {

    // ==========================================
    // TEST 1: METRIC COMPARISON AND CSV EXPORT
    // ==========================================

    @Test
    void comparisonExportsCorrectMetricDeltas(
            @TempDir Path tempDir) throws Exception {

        ImprovementResult baseline =
                new ImprovementResult("J48", "Baseline");

        baseline.setAccuracy(75.0);
        baseline.setPrecision(Double.NaN);
        baseline.setRecall(0.06);
        baseline.setF1Score(Double.NaN);
        baseline.setRocAuc(0.50);

        ImprovementResult improved =
                new ImprovementResult(
                        "J48", "YesWeight=4"
                );

        improved.setAccuracy(66.5);
        improved.setPrecision(0.22);
        improved.setRecall(0.25);
        improved.setF1Score(0.23);
        improved.setRocAuc(0.51);

        ModelComparator.Comparison comparison =
                ModelComparator.compare(
                        baseline, improved
                );

        Path csv = tempDir.resolve("comparison.csv");

        ModelComparator.saveCsv(
                List.of(comparison),
                csv.toString()
        );

        assertTrue(Files.exists(csv));

        List<String> lines = Files.readAllLines(csv);

        // Header + one comparison row
        assertEquals(2, lines.size());

        String row = lines.get(1);

        // Accuracy delta = 66.5 - 75.0 = -8.5 pp
        assertTrue(
                row.contains("75.0000,66.5000,-8.5000")
        );

        // Recall delta = 0.25 - 0.06 = +0.19
        assertTrue(
                row.contains("0.0600,0.2500,0.1900")
        );

        // Undefined precision must remain N/A
        assertTrue(
                row.contains("N/A,0.2200,N/A")
        );
    }

    // ==========================================
    // TEST 2: REJECT DIFFERENT MODEL NAMES
    // ==========================================

    @Test
    void comparisonRejectsDifferentModelNames() {

        ImprovementResult baseline =
                new ImprovementResult("J48", "Baseline");

        ImprovementResult improved =
                new ImprovementResult(
                        "RandomForest", "Weighted"
                );

        assertThrows(
                IllegalArgumentException.class,
                () -> ModelComparator.compare(
                        baseline, improved
                )
        );
    }
}
