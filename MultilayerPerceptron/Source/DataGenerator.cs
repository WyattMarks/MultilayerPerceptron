using System;
using System.Collections.Generic;
using System.Text;

namespace MultilayerPerceptron {
    class DataGenerator { //Generate the half-moon data set
        private Random rand;

        private double radius; //Radius from the midpoint to the middle of the outline
        private double width; //Width of the outline
        private double distance; //Distance between each set (red, blue)

        private int samples; //Number of DataPoints to generate

        public DataGenerator(int samples, double radius, double width, double distance) {
            this.samples = samples;
            this.radius = radius;
            this.width = width;
            this.distance = distance;

            rand = new Random(DateTime.Now.Millisecond);
        }

        public DataPoint[] GenerateData() {
            DataPoint[] points = new DataPoint[samples];

            for (int i = 0; i < samples / 2; i++) { //Generate "red" data
                double radius = this.radius - width / 2 + width * rand.NextDouble();
                double theta = Math.PI * rand.NextDouble();
                double x = radius * Math.Cos(theta);
                double y = radius * Math.Sin(theta);


                points[i] = new DataPoint(x, y, 1);
            }

            for (int i = samples / 2; i < samples; i++) { //Generate "blue" data
                double radius = this.radius - width / 2 + width * rand.NextDouble();
                double theta = Math.PI * rand.NextDouble();
                double x = radius * Math.Cos(-theta) + this.radius;
                double y = radius * Math.Sin(-theta) - this.distance;

                points[i] = new DataPoint(x, y, -1);
            }

            return points;
        }
    }
}
