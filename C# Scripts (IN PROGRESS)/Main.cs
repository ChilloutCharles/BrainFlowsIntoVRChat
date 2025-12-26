using System;
using System.Numerics;
using System.Runtime.Serialization;

using brainflow;
using brainflow.math;

class GetBoardData
    {
        static void Main(string[] args)
        {
            BoardShim.enable_dev_board_logger();

            BrainFlowInputParams input_params = new BrainFlowInputParams();
            int board_id = parse_args(args, input_params);

            BoardShim board_shim = new BoardShim(board_id, input_params);
            board_shim.prepare_session();
            board_shim.start_stream();
            System.Threading.Thread.Sleep(5000);
            board_shim.stop_stream();
            double[,] unprocessed_data = board_shim.get_current_board_data(20);
            int[] eeg_channels = BoardShim.get_eeg_channels(board_id);
            foreach (var index in eeg_channels)
                Console.WriteLine("[{0}]", string.Join(", ", unprocessed_data.GetRow(index)));
            board_shim.release_session();
        }

        static int parse_args(string[] args, BrainFlowInputParams input_params)
        {
            int board_id = (int)BoardIds.SYNTHETIC_BOARD; //assume synthetic board by default
            // use docs to get params for your specific board, e.g. set serial_port for Cyton
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i].Equals("--ip-address"))
                {
                    input_params.ip_address = args[i + 1];
                }
                if (args[i].Equals("--mac-address"))
                {
                    input_params.mac_address = args[i + 1];
                }
                if (args[i].Equals("--serial-port"))
                {
                    input_params.serial_port = args[i + 1];
                }
                if (args[i].Equals("--other-info"))
                {
                    input_params.other_info = args[i + 1];
                }
                if (args[i].Equals("--ip-port"))
                {
                    input_params.ip_port = Convert.ToInt32(args[i + 1]);
                }
                if (args[i].Equals("--ip-protocol"))
                {
                    input_params.ip_protocol = Convert.ToInt32(args[i + 1]);
                }
                if (args[i].Equals("--board-id"))
                {
                    board_id = Convert.ToInt32(args[i + 1]);
                }
                if (args[i].Equals("--timeout"))
                {
                    input_params.timeout = Convert.ToInt32(args[i + 1]);
                }
                if (args[i].Equals("--serial-number"))
                {
                    input_params.serial_number = args[i + 1];
                }
                if (args[i].Equals("--file"))
                {
                    input_params.file = args[i + 1];
                }
            }
            return board_id;
        }
}
class Markers
{
    static void Main(string[] args)
    {
        BoardShim.enable_dev_board_logger();

        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = parse_args(args, input_params);

        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream();
        board_shim.add_streamer("file://data.csv:w");
        for (int i = 1; i < 5; i++)
        {
            System.Threading.Thread.Sleep(1000);
            board_shim.insert_marker(i);
        }
        board_shim.stop_stream();
        board_shim.release_session();
    }

    static int parse_args(string[] args, BrainFlowInputParams input_params)
    {
        int board_id = (int)BoardIds.SYNTHETIC_BOARD; //assume synthetic board by default
                                                      // use docs to get params for your specific board, e.g. set serial_port for Cyton
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals("--ip-address"))
            {
                input_params.ip_address = args[i + 1];
            }
            if (args[i].Equals("--mac-address"))
            {
                input_params.mac_address = args[i + 1];
            }
            if (args[i].Equals("--serial-port"))
            {
                input_params.serial_port = args[i + 1];
            }
            if (args[i].Equals("--other-info"))
            {
                input_params.other_info = args[i + 1];
            }
            if (args[i].Equals("--ip-port"))
            {
                input_params.ip_port = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--ip-protocol"))
            {
                input_params.ip_protocol = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--board-id"))
            {
                board_id = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--timeout"))
            {
                input_params.timeout = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--serial-number"))
            {
                input_params.serial_number = args[i + 1];
            }
            if (args[i].Equals("--file"))
            {
                input_params.file = args[i + 1];
            }
        }
        return board_id;
    }
}
class Serialization
{
    static void Main(string[] args)
    {
        // use synthetic board for demo
        BoardShim.enable_dev_board_logger();
        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = (int)BoardIds.SYNTHETIC_BOARD;

        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(5000);
        board_shim.stop_stream();
        double[,] unprocessed_data = board_shim.get_current_board_data(20);
        int[] eeg_channels = BoardShim.get_eeg_channels(board_id);
        Console.WriteLine("Before serialization:");
        foreach (var index in eeg_channels)
            Console.WriteLine("[{0}]", string.Join(", ", unprocessed_data.GetRow(index)));
        board_shim.release_session();

        // demo for data serialization
        DataFilter.write_file(unprocessed_data, "test.csv", "w");
        double[,] restored_data = DataFilter.read_file("test.csv");
        Console.WriteLine("After Serialization:");
        foreach (var index in eeg_channels)
            Console.WriteLine("[{0}]", string.Join(", ", restored_data.GetRow(index)));
    }
}
class Downsampling
{
    static void Main(string[] args)
    {
        // use synthetic board for demo
        BoardShim.enable_dev_board_logger();
        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = (int)BoardIds.SYNTHETIC_BOARD;

        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(5000);
        board_shim.stop_stream();
        double[,] unprocessed_data = board_shim.get_board_data();
        int[] eeg_channels = BoardShim.get_eeg_channels(board_id);
        board_shim.release_session();

        for (int i = 0; i < eeg_channels.Length; i++)
        {
            Console.WriteLine("Before processing:");
            Console.WriteLine("[{0}]", string.Join(", ", unprocessed_data.GetRow(eeg_channels[i])));
            // you can use MEAN, MEDIAN or EACH for downsampling
            double[] filtered = DataFilter.perform_downsampling(unprocessed_data.GetRow(eeg_channels[i]), 3, (int)AggOperations.MEDIAN);
            Console.WriteLine("Before processing:");
            Console.WriteLine("[{0}]", string.Join(", ", filtered));
        }
    }
}
class Transforms
{
    static void Main(string[] args)
    {
        // use synthetic board for demo
        BoardShim.enable_dev_board_logger();
        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = (int)BoardIds.SYNTHETIC_BOARD;

        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(5000);
        board_shim.stop_stream();
        double[,] unprocessed_data = board_shim.get_current_board_data(64);
        int[] eeg_channels = BoardShim.get_eeg_channels(board_id);
        board_shim.release_session();

        for (int i = 0; i < eeg_channels.Length; i++)
        {
            Console.WriteLine("Original data:");
            Console.WriteLine("[{0}]", string.Join(", ", unprocessed_data.GetRow(eeg_channels[i])));
            // demo for wavelet transform
            // tuple of coeffs array in format[A(J) D(J) D(J-1) ..... D(1)] where J is a
            // decomposition level, A - app coeffs, D - detailed coeffs, and array which stores
            // length for each block, len of this array is decomposition_length + 1
            Tuple<double[], int[]> wavelet_data = DataFilter.perform_wavelet_transform(unprocessed_data.GetRow(eeg_channels[i]), (int)WaveletTypes.DB4, 1, (int)WaveletExtensionTypes.SYMMETRIC);
            // print app coeffs
            for (int j = 0; j < wavelet_data.Item2[0]; j++)
            {
                Console.Write(wavelet_data.Item1[j] + " ");
            }
            Console.WriteLine();
            // you can do smth with wavelet coeffs here, for example denoising works via thresholds for wavelets coeffs
            double[] restored_data = DataFilter.perform_inverse_wavelet_transform(wavelet_data, unprocessed_data.GetRow(eeg_channels[i]).Length, (int)WaveletTypes.DB4, 1, (int)WaveletExtensionTypes.SYMMETRIC);
            Console.WriteLine("Restored wavelet data:");
            Console.WriteLine("[{0}]", string.Join(", ", restored_data));

            // demo for fft
            // end_pos - start_pos must be a power of 2
            Complex[] fft_data = DataFilter.perform_fft(unprocessed_data.GetRow(eeg_channels[i]), 0, 64, (int)WindowOperations.HAMMING);
            // len of fft_data is N / 2 + 1
            double[] restored_fft_data = DataFilter.perform_ifft(fft_data);
            Console.WriteLine("Restored fft data:");
            Console.WriteLine("[{0}]", string.Join(", ", restored_fft_data));
        }
    }
}
class SignalFiltering
{
    static void Main(string[] args)
    {
        // use synthetic board for demo
        BoardShim.enable_dev_board_logger();
        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = (int)BoardIds.SYNTHETIC_BOARD;

        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(5000);
        board_shim.stop_stream();
        double[,] unprocessed_data = board_shim.get_current_board_data(20);
        int[] eeg_channels = BoardShim.get_eeg_channels(board_id);
        board_shim.release_session();

        for (int i = 0; i < eeg_channels.Length; i++)
        {
            DataFilter.detrend(unprocessed_data, eeg_channels[i], (int)DetrendOperations.CONSTANT);
            DataFilter.perform_bandstop(unprocessed_data, eeg_channels[i], BoardShim.get_sampling_rate(board_id), 48.0, 52.0, 4, (int)FilterTypes.BUTTERWORTH, 0.0);
            DataFilter.perform_bandpass(unprocessed_data, eeg_channels[i], BoardShim.get_sampling_rate(board_id), 4.0, 30.0, 4, (int)FilterTypes.BUTTERWORTH, 0.0);
        }
    }
}
class Denoising
{
    static void Main(string[] args)
    {
        // use synthetic board for demo
        BoardShim.enable_dev_board_logger();
        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = (int)BoardIds.SYNTHETIC_BOARD;

        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(5000);
        board_shim.stop_stream();
        double[,] unprocessed_data = board_shim.get_current_board_data(64);
        int[] eeg_channels = BoardShim.get_eeg_channels(board_id);
        foreach (var index in eeg_channels)
            Console.WriteLine("[{0}]", string.Join(", ", unprocessed_data.GetRow(index)));
        board_shim.release_session();

        // for demo apply different methods to different channels
        double[] filtered;
        for (int i = 0; i < eeg_channels.Length; i++)
        {
            switch (i)
            {
                // first of all you can try simple moving average or moving median
                case 0:
                    filtered = DataFilter.perform_rolling_filter(unprocessed_data.GetRow(eeg_channels[i]), 3, (int)AggOperations.MEAN);
                    Console.WriteLine("Filtered channel " + eeg_channels[i]);
                    Console.WriteLine("[{0}]", string.Join(", ", filtered));
                    break;
                case 1:
                    filtered = DataFilter.perform_rolling_filter(unprocessed_data.GetRow(eeg_channels[i]), 3, (int)AggOperations.MEDIAN);
                    Console.WriteLine("Filtered channel " + eeg_channels[i]);
                    Console.WriteLine("[{0}]", string.Join(", ", filtered));
                    break;
                // if for your signal these methods dont work good you can try wavelet based denoising
                default:
                    // feel free to try different functions and different decomposition levels
                    filtered = DataFilter.perform_wavelet_denoising(unprocessed_data.GetRow(eeg_channels[i]), (int)WaveletTypes.BIOR3_9, 3);
                    Console.WriteLine("Filtered channel " + eeg_channels[i]);
                    Console.WriteLine("[{0}]", string.Join(", ", filtered));
                    break;
            }
        }
    }
}
class BandPower
{
    static void Main(string[] args)
    {
        // use synthetic board for demo
        BoardShim.enable_dev_board_logger();
        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = (int)BoardIds.SYNTHETIC_BOARD;
        BoardDescr board_descr = BoardShim.get_board_descr<BoardDescr>(board_id);
        int sampling_rate = board_descr.sampling_rate;
        int nfft = DataFilter.get_nearest_power_of_two(sampling_rate);

        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(10000);
        board_shim.stop_stream();
        double[,] data = board_shim.get_board_data();
        int[] eeg_channels = board_descr.eeg_channels;
        // use second channel of synthetic board to see 'alpha'
        int channel = eeg_channels[1];
        board_shim.release_session();
        double[] detrend = DataFilter.detrend(data.GetRow(channel), (int)DetrendOperations.LINEAR);
        Tuple<double[], double[]> psd = DataFilter.get_psd_welch(detrend, nfft, nfft / 2, sampling_rate, (int)WindowOperations.HANNING);
        double band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0);
        double band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0);
        Console.WriteLine("Alpha/Beta Ratio:" + (band_power_alpha / band_power_beta));
    }
}
class EEGMetrics
{
    static void Main(string[] args)
    {
        // use synthetic board for demo
        BoardShim.enable_dev_board_logger();
        BrainFlowInputParams input_params = new BrainFlowInputParams();
        int board_id = parse_args(args, input_params);
        BoardShim board_shim = new BoardShim(board_id, input_params);
        int sampling_rate = BoardShim.get_sampling_rate(board_shim.get_board_id());
        int[] eeg_channels = BoardShim.get_eeg_channels(board_shim.get_board_id());

        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(10000);
        board_shim.stop_stream();
        double[,] data = board_shim.get_board_data();
        board_shim.release_session();

        Tuple<double[], double[]> bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, true);
        double[] feature_vector = bands.Item1;
        BrainFlowModelParams model_params = new BrainFlowModelParams((int)BrainFlowMetrics.MINDFULNESS, (int)BrainFlowClassifiers.DEFAULT_CLASSIFIER);
        MLModel model = new MLModel(model_params);
        model.prepare();
        Console.WriteLine("Score: " + model.predict(feature_vector)[0]);
        model.release();
    }

    static int parse_args(string[] args, BrainFlowInputParams input_params)
    {
        int board_id = (int)BoardIds.SYNTHETIC_BOARD; //assume synthetic board by default
                                                      // use docs to get params for your specific board, e.g. set serial_port for Cyton
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals("--ip-address"))
            {
                input_params.ip_address = args[i + 1];
            }
            if (args[i].Equals("--mac-address"))
            {
                input_params.mac_address = args[i + 1];
            }
            if (args[i].Equals("--serial-port"))
            {
                input_params.serial_port = args[i + 1];
            }
            if (args[i].Equals("--other-info"))
            {
                input_params.other_info = args[i + 1];
            }
            if (args[i].Equals("--ip-port"))
            {
                input_params.ip_port = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--ip-protocol"))
            {
                input_params.ip_protocol = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--board-id"))
            {
                board_id = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--timeout"))
            {
                input_params.timeout = Convert.ToInt32(args[i + 1]);
            }
            if (args[i].Equals("--serial-number"))
            {
                input_params.serial_number = args[i + 1];
            }
            if (args[i].Equals("--file"))
            {
                input_params.file = args[i + 1];
            }
        }
        return board_id;
    }
}

class ICA
{
    static void Main(string[] args)
    {
        BoardShim.enable_dev_board_logger();

        int board_id = (int)BoardIds.SYNTHETIC_BOARD;
        BoardDescr board_descr = BoardShim.get_board_descr<BoardDescr>(board_id);
        int[] eeg_channels = board_descr.eeg_channels;
        int channel = eeg_channels[1];

        BrainFlowInputParams input_params = new BrainFlowInputParams();
        BoardShim board_shim = new BoardShim(board_id, input_params);
        board_shim.prepare_session();
        board_shim.start_stream(3600);
        System.Threading.Thread.Sleep(10000);
        board_shim.stop_stream();
        double[,] data = board_shim.get_board_data(500);
        board_shim.release_session();

        double[,] ica_data = data.GetRow(channel).Reshape(5, 100);
        Tuple<double[,], double[,], double[,], double[,]> ica = DataFilter.perform_ica(ica_data, 2);
    }
}