using System;
using System.Linq;

public static class BrainFlowUtils
{
    // --- Manual Butterworth Lowpass (2nd Order) ---
    public static double[] ManualLowPass(double[] data, double samplingRate, double cutoff = 10.0)
    {
        int n = data.Length;
        double[] output = new double[n];

        // Calculate coefficients
        double wc = Math.Tan(Math.PI * cutoff / samplingRate);
        double k1 = 1.41421356237 * wc; // sqrt(2) * wc
        double k2 = wc * wc;
        double a = k2 / (1 + k1 + k2);
        double b = 2 * a;
        double c = a;
        double d = 2 * (k2 - 1) / (1 + k1 + k2);
        double e = (1 - k1 + k2) / (1 + k1 + k2);

        // Filter state variables
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;

        for (int i = 0; i < n; i++)
        {
            double x0 = data[i];
            double y0 = a * x0 + b * x1 + c * x2 - d * y1 - e * y2;

            output[i] = y0;

            // Update state
            x2 = x1; x1 = x0;
            y2 = y1; y1 = y0;
        }
        return output;
    }

    public static bool[] GetArtifactMask(double[] data, double samplingRate, double stdMult = 4.0, bool isAbsolute = true)
    {
        // 1. Filter the data (10Hz Lowpass)
        double[] filtered = ManualLowPass(data, samplingRate, 10.0);

        // 2. Manual Mean
        double sum = 0;
        foreach (var v in filtered) sum += v;
        double mean = sum / filtered.Length;

        // 3. Manual Standard Deviation
        double sumSq = 0;
        foreach (var v in filtered) sumSq += Math.Pow(v - mean, 2);
        double std = Math.Sqrt(sumSq / filtered.Length);

        double threshold = stdMult * std;
        bool[] mask = new bool[data.Length];

        // 4. Thresholding
        for (int i = 0; i < filtered.Length; i++)
        {
            double diff = filtered[i] - mean;
            if (isAbsolute) diff = Math.Abs(diff);

            mask[i] = diff > threshold;
        }

        return mask;
    }
}