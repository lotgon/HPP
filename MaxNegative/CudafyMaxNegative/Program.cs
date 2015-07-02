using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using System.IO;

using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Diagnostics;


namespace CudafyMaxNegative
{
    class Program
    {
        static void Main(string[] args)
        {
            List<double> pricesList = new List<double>();
            List<string> datesList = new List<string>();

            using( var streamReader = new System.IO.StreamReader(Settings1.Default.InputCsvFileName) )
            {
                CsvReader csv = new CsvReader(streamReader);
                csv.Configuration.HasHeaderRecord = true;
                while (csv.Read())
                {
                    string header = csv.GetField<string>(0);
                    double price = csv.GetField<double>(1);
                    string dataTime = csv.GetField<string>(2);

                    pricesList.Add(price);
                    datesList.Add(dataTime);
                }
            }

            using( StreamWriter sw = new StreamWriter(Settings1.Default.InputCsvFileName+"_Cudafy.csv"))
            {
                MeasureTime("GPGPU seq", 1, ()=>
                    {
                        CalculateAll(null, pricesList.ToArray(), datesList.ToArray(), 0.5f, 2, 0.1);
                    });
            }

        }

        static GPGPU InitGPU()
        {
            CudafyModes.Target = eGPUType.OpenCL; // To use OpenCL, change this enum
            CudafyModes.DeviceId = 0;
            CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;

            int deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
            if (deviceCount == 0)
            {
                Console.WriteLine("No suitable {0} devices found.", CudafyModes.Target);
            }
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            Console.WriteLine("Running examples using {0}", gpu.GetDeviceProperties(false).Name);

            //CudafyTranslator.GenerateDebug = true;
            CudafyModule km = CudafyTranslator.Cudafy();
            gpu.LoadModule(km);

            return gpu;
        }


        static void CalculateAll(StreamWriter sq, double[] prices, string[] dates, double startTp, double endTp, double stepTp)
        {
            GPGPU gpu = InitGPU();

            int N = prices.Length;
            double[] drawdown = new double[N];
            int[] duration = new int[N];

            double[] dev_prices = gpu.Allocate<double>(prices);
            double[] dev_drawdown = gpu.Allocate<double>(drawdown);
            int[] dev_duration= gpu.Allocate<int>(duration);

            gpu.CopyToDevice(prices, dev_prices);

            for (double currTp = startTp; currTp <= endTp; currTp += stepTp)
            {
                int BLOCK_SIZE = 32;
                gpu.Launch((N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE).calculate(dev_prices, currTp, N, dev_drawdown, dev_duration);

                gpu.CopyFromDevice<double>(dev_drawdown, drawdown);
                gpu.CopyFromDevice<int>(dev_duration, duration);

                if( sq!= null)
                {
                    for( int i=0;i<N;i++)
                    {
                        sq.WriteLine(string.Format("{0}, {1}, {2}, {3}", dates[i], currTp, Math.Round(drawdown[i], 6), duration[i]));
                    }
                }
            }
            gpu.FreeAll();
        }
        [Cudafy]
        public static void calculate(GThread gt, double[] prices, double currTp, int N, double[] drawdownArray, int[] durationArray)
        {
            int absTx = gt.blockIdx.x * gt.blockDim.x + gt.threadIdx.x;

            if (absTx >= N)
                return;

            //Console.WriteLine("blockIdx.x = " + gt.blockIdx.x.ToString());//  blockDim.x={1} threadIdx.x={2}", gt.blockDim.x, gt.threadIdx.x));
            double threshold = prices[absTx] + currTp;
            double openPrice = prices[absTx];
            double drawdown = 0;
            int step;
            for (step = absTx; step < N && prices[step] < threshold ; step++)
            {
                if (openPrice - prices[step] > drawdown)
                    drawdown = openPrice - prices[step];
            }
            if (step < N)
                durationArray[absTx] = step - absTx;
            else
                durationArray[absTx] = -1;
            drawdownArray[absTx] = drawdown;

        }

        internal static void MeasureSpeed(string name, int times, Action testAction, Func<double, string> addInfo = null)
        {
            Stopwatch watch = Stopwatch.StartNew();
            for (int i = 0; i < times; i++)
                testAction();
            watch.Stop();
            double callsPerSecond = times * 1000.0 / (watch.ElapsedMilliseconds);
            string output = "\t" + name + " calls per second=> " + callsPerSecond + ";";
            if (addInfo != null)
                output += addInfo(callsPerSecond);
            Console.WriteLine(output);
        }
        internal static void MeasureTime(string name, int times, Action testAction, Func<double, string> addInfo = null)
        {
            Stopwatch watch = Stopwatch.StartNew();
            for (int i = 0; i < times; i++)
                testAction();
            watch.Stop();
            string output = "\t" + name + " took=> " + watch.ElapsedMilliseconds + " millisencods;";
            Console.WriteLine(output);
        }
    }
}
