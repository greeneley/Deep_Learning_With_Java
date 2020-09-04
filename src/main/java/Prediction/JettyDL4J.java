package Prediction;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.handler.AbstractHandler;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class JettyDL4J extends AbstractHandler {
    /**
     * the model loaded from Keras
     **/
    private MultiLayerNetwork model;

    /**
     * the number of input parameters in the Keras model
     **/
    private static int inputs = 10;

    /**
     * launch a web server on port 8080
     */
    public static void main(String[] args) throws Exception {
        Server server = new Server(8081);
        server.setHandler(new JettyDL4J());
        server.start();
        server.join();
    }

    /**
     * Loads the Keras Model
     **/
    public JettyDL4J() throws Exception {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));
        String simpleMlp = new ClassPathResource("games.h5").getFile().getPath();
        model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);
    }

    /**
     * Returns a prediction for the passed in data set
     **/
    public void handle(String target, Request baseRequest, HttpServletRequest request,
                       HttpServletResponse response) throws IOException, ServletException {

        // create a dataset from the input parameters
        INDArray features = Nd4j.zeros(inputs);
        for (int i = 0; i < inputs; i++) {
            features.putScalar(new int[]{i}, Double.parseDouble(baseRequest.getParameter("G" + (i + 1))));
        }

        // output the estimate
        double prediction = model.output(features).getDouble(0);
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println("Prediction: " + prediction);
        baseRequest.setHandled(true);
    }
}
