using System;
using System.Linq;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MultilayerPerceptron  {
    class Perceptron {
        private DataPoint[] data;

        private DataPoint[] trainingData;

        public double[][][] weights;
        public double[][][] deltaweights;
        public double[][][] initialWeights;

        public double[] MSE;


        double learningRate = 0.1;
        double maxRate = 0.1;
        double minRate = 0.00001;

        int n_epochs = 40;
        double mse_thresh = 0.001;
        int n_layers = 2;
        int n_input = 3;
        int n_hd = 20;
        int n_output = 1;

        private Random rand;
        public Perceptron(DataPoint[] data, double trainToTestRatio = 0.2) { // 1/5 of the data should be used for training (500 if 2500 samples)
            rand = new Random(DateTime.Now.Millisecond);

            DataPoint[] shuffledData = data.OrderBy(x => rand.Next()).ToArray(); //Shuffles the data

            this.data = shuffledData;

            NormalizeData();

            int num_tr = (int)(data.Length * trainToTestRatio);

            trainingData = new DataPoint[num_tr];

            for (int i = 0; i < num_tr; i++) {
                trainingData[i] = this.data[i];
            }


            //Construct MLP

            weights = new double[n_layers][][];
            deltaweights = new double[n_layers][][];

            weights[0] = new double[n_hd][];
            deltaweights[0] = new double[n_hd][];
            for (int i = 0; i < n_hd; i++) {
                weights[0][i] = new double[n_input];
                deltaweights[0][i] = new double[n_input];
                for (int j = 0; j < n_input; j++) {
                    weights[0][i][j] = rand.NextDouble() * 2 - 1;
                    deltaweights[0][i][j] = 0;
                }
            }

            weights[1] = new double[n_output][];
            deltaweights[1] = new double[n_output][];
            for (int i = 0; i < n_output; i++) {
                weights[1][i] = new double[n_hd + 1];
                deltaweights[1][i] = new double[n_hd + 1];
                for (int j = 0; j < n_hd + 1; j++) {
                    weights[1][i][j] = rand.NextDouble() * 2 - 1;
                    deltaweights[1][i][j] = 0;
                }
            }


            initialWeights = new double[weights.Length][][];
            for (int i = 0; i < weights.Length; i++) {
                initialWeights[i] = new double[weights[i].Length][];
                for (int j = 0; j < weights[i].Length; j++) {
                    initialWeights[i][j] = (double[])weights[i][j].Clone();
                }
            }
        }

        private void NormalizeData() {
            DataPoint average = new DataPoint(0, 0, 1);
            for (int i = 0; i < data.Length; i++) {
                average.x += data[i].x;
                average.y += data[i].y;
            }
            average.x /= data.Length;
            average.y /= data.Length;

            DataPoint max = new DataPoint(0, 0, 1);
            for (int i = 0; i < data.Length; i++) {
                data[i].x -= average.x;
                data[i].y -= average.y;

                if (data[i].x > max.x)
                    max.x = data[i].x;
                if (data[i].y > max.y)
                    max.y = data[i].y;
            }

            for (int i = 0; i < data.Length; i++) {
                data[i].x /= max.x;
                data[i].y /= max.y;
            }

        }


        public void AnnealLearningRate() {
            double step = (maxRate - minRate) / ( n_epochs - 1 );
            learningRate -= step;
        }


        public void Train() {
            int curEpoch = 0;
            MSE = new double[n_epochs];
            MSE[0] = double.MaxValue;

            while (curEpoch < n_epochs && MSE[curEpoch] > mse_thresh) {

                double[] e = new double[trainingData.Length];
                double squareErrorSum = 0;

                for (int i = 0; i < trainingData.Length; i++) {

                    //Forward Computation

                    double[] point = new double[3];
                    point[0] = 1;
                    point[1] = trainingData[i].x;
                    point[2] = trainingData[i].y;
                    int d = trainingData[i].label;

                    double[] y = new double[n_hd + 1];
                    y[0] = 1;

                    for (int j = 1; j < n_hd + 1; j++) {
                        y[j] = ActivationFunction(Neuron(point, weights[0][j - 1]));
                    }

                    double v2 = Neuron(y, weights[1][0]);
                    double output = ActivationFunction(v2);

                    e[i] = (double) d - output;
                    squareErrorSum += e[i] * e[i];

                    //Backward computation

                    //eta * error * -1 * phiprime(v2) * y
                    double factor = learningRate * e[i] * ActivationDerivative(v2);
                    for (int j = 0; j < n_hd + 1; j++) {
                        deltaweights[1][0][j] = factor * y[j];
                        weights[1][0][j] += deltaweights[1][0][j];
                    }


                    factor = learningRate * e[i] * ActivationDerivative(v2);
                    for (int j = 0; j < n_hd; j++) {
                        for (int k = 0; k < n_input; k++) {
                            deltaweights[0][j][k] = factor * weights[1][0][j] * ActivationDerivative(Neuron(point, weights[0][j])) * point[k];
                            weights[0][j][k] += deltaweights[0][j][k];
                        }
                    }


                }

                MSE[curEpoch] = squareErrorSum / trainingData.Length;
                Trace.WriteLine($"Epoch:{curEpoch+1}/{n_epochs}, MSE: {MSE[curEpoch]} / {mse_thresh}, ETA: {learningRate}");

                curEpoch++;
                AnnealLearningRate();
                if (curEpoch < n_epochs)
                    MSE[curEpoch] = double.MaxValue;
            }
        }

        public DataPoint[] Test(double[][][] w) {
            DataPoint[] output = new DataPoint[data.Length];

            for (int n = 0; n < data.Length; n++) {
                double[] x = new double[3];
                x[0] = 1;
                x[1] = data[n].x;
                x[2] = data[n].y;

                double[] y = new double[n_hd + 1];
                y[0] = 1;

                for (int j = 1; j < n_hd + 1; j++) {
                    y[j] = ActivationFunction(Neuron(x, w[0][j - 1]));
                }

                double v2 = Neuron(y, w[1][0]);
                double o = ActivationFunction(v2);

                if (o > 0) {
                    o = 1;
                } else {
                    o = -1;
                }


                output[n] = new DataPoint(x[1], x[2], (int)o); //Basically the same point as data[n], just with a label generated by our perceptron!
            }


            return output;
        }

        public double Neuron(double[] x, double[] w) {
            double v = 0;

            for (int i = 0; i < x.Length; i++) {
                v += x[i] * w[i];
            }

            return v;
        }

        public double ActivationFunction(double v) {
            return ActivationFunctions.HyperbolicActivation(v);
        }

        public double ActivationDerivative(double v) {
            return ActivationFunctions.HyperbolicDerivative(v);
        }
    }
}
