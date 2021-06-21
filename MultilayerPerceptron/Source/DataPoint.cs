namespace MultilayerPerceptron {
    class DataPoint {
        public double x;
        public double y;
        public int label;

        public DataPoint(double x, double y, int label) {
            this.x = x;
            this.y = y;
            this.label = label;
        }
    }
}
