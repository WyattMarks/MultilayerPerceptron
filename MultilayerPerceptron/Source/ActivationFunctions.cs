using System;

namespace MultilayerPerceptron {
    public static class ActivationFunctions {
        public static double SigmoidActivation(double v) {
            return 1.0 / (1.0 + Math.Exp(-v));
        }

        public static double ThresholdActivation(double v) {
            if (v >= 0) {
                return 1;
            } else {
                return -1;
            }
        }

        public static double HyperbolicActivation(double v) {
            double e2v = Math.Exp(2 * v);
            return (e2v - 1.0) / (e2v + 1.0);
        }

        public static double HyperbolicDerivative(double v) {
            double hyper = HyperbolicActivation(v);
            return (1.0 - hyper * hyper);
        }

        public static double ReLuActivation(double v) {
            return Math.Max(0, v);
        }
    }
}
