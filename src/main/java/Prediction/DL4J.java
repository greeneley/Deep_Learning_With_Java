package Prediction;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class DL4J {
    public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        String simpleMlp = new ClassPathResource("games.h5").getFile().getPath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);

        int inputs = 10;
        INDArray features = Nd4j.zeros(inputs);
        for (int i = 0; i < inputs; i++)
            features.putScalar(new int[]{i}, Math.random() < 0.5 ? 0 : 1);

        double prediction = model.output(features).getDouble(0);
    }
}
