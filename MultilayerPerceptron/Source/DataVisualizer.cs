namespace MultilayerPerceptron {
    using System;
    using OxyPlot;
    using OxyPlot.Axes;
    using OxyPlot.Series;
    public class DataVisualizer {
        public DataVisualizer() {

            //Visualize the initial Training / Test Data
            this.TestDataModel = new PlotModel { Title = "Training and Test Data" };

            this.TestDataModel.Axes.Add(new LinearColorAxis { Position = AxisPosition.None, Minimum = 0.1, Maximum = 0.9, HighColor = OxyColors.Red, LowColor = OxyColors.Blue });

            this.TestDataModel.Axes.Add(new LinearAxis() { Title = "X2", Position = AxisPosition.Left });
            this.TestDataModel.Axes.Add(new LinearAxis() { Title = "X1", Position = AxisPosition.Bottom });

            var dataPoints = new DataGenerator(2500, 10, 6, -5).GenerateData(); //Generate the data 

            var series = new ScatterSeries { MarkerType = MarkerType.Star };

            for (int i = 0; i < 2500; i++) {
                series.Points.Add(new ScatterPoint(dataPoints[i].x, dataPoints[i].y) { Value = dataPoints[i].label });
            }

            this.TestDataModel.Series.Add(series);

            Perceptron test = new Perceptron(dataPoints);
            test.Train();





            this.FinalWeightsModel = new PlotModel { Title = "Final Weights" };

            this.FinalWeightsModel.Axes.Add(new LinearColorAxis { Position = AxisPosition.None, Minimum = 0.1, Maximum = 0.9, HighColor = OxyColors.Red, LowColor = OxyColors.Blue });

            this.FinalWeightsModel.Axes.Add(new LinearAxis() { Title = "X2", Position = AxisPosition.Left });
            this.FinalWeightsModel.Axes.Add(new LinearAxis() { Title = "X1", Position = AxisPosition.Bottom });

            dataPoints = test.Test(test.weights); //Generate the labelled data using the final weights

            series = new ScatterSeries { MarkerType = MarkerType.Star };

            for (int i = 0; i < 2500; i++) {
                var color = (dataPoints[i].label == 1) ? 1 : 0;
                series.Points.Add(new ScatterPoint(dataPoints[i].x, dataPoints[i].y) { Value = color });
            }

            this.FinalWeightsModel.Series.Add(series);


            this.MSEModel = new PlotModel { Title = "Mean Square Error" };

            this.MSEModel.Axes.Add(new LinearColorAxis { Position = AxisPosition.None, Minimum = 0.1, Maximum = 0.9, HighColor = OxyColors.Red, LowColor = OxyColors.Blue });

            this.MSEModel.Axes.Add(new LinearAxis() { Title = "Mean Square Error", Position = AxisPosition.Left });
            this.MSEModel.Axes.Add(new LinearAxis() { Title = "Training Iteration", Position = AxisPosition.Bottom });

            series = new ScatterSeries { MarkerType = MarkerType.Circle };
            var errorSeries = new LineSeries();

            for (int i = 0; i < test.MSE.Length; i++) {
                errorSeries.Points.Add(new OxyPlot.DataPoint(i, test.MSE[i]));
                series.Points.Add(new ScatterPoint(i, test.MSE[i]) { Value = Math.Min(1, test.MSE[i] / test.MSE[0]) });
            }

            this.MSEModel.Series.Add(series);
            this.MSEModel.Series.Add(errorSeries);


            this.InitialWeightsModel = new PlotModel { Title = "Initial Weights" };

            this.InitialWeightsModel.Axes.Add(new LinearColorAxis { Position = AxisPosition.None, Minimum = 0.1, Maximum = 0.9, HighColor = OxyColors.Red, LowColor = OxyColors.Blue });

            this.InitialWeightsModel.Axes.Add(new LinearAxis() { Title = "X2", Position = AxisPosition.Left });
            this.InitialWeightsModel.Axes.Add(new LinearAxis() { Title = "X1", Position = AxisPosition.Bottom });

            dataPoints = test.Test(test.initialWeights); //Generate the labelled data using the initial weights

            series = new ScatterSeries { MarkerType = MarkerType.Star };

            for (int i = 0; i < 2500; i++) {
                series.Points.Add(new ScatterPoint(dataPoints[i].x, dataPoints[i].y) { Value = dataPoints[i].label });
            }

            this.InitialWeightsModel.Series.Add(series);
        }


            

        public PlotModel TestDataModel { get; private set; }
        public PlotModel InitialWeightsModel { get; private set; }
        public PlotModel FinalWeightsModel { get; private set; }
        public PlotModel MSEModel { get; private set; }
    }
}