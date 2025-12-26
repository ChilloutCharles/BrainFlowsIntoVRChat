using System;

namespace BrainFlowInterface
{
    public static class BFIConstants
    {
        public const int VersionMajor = 1;
        public const int VersionMinor = 0;

        public const string OscBasePath = "/avatar/parameters/";
        public const string BfiRoot = "BFI";

        public static readonly string FullRootPath = $"{OscBasePath}{BfiRoot}";
    }
    public enum BandPowers
    {
        Delta = 0,
        Theta = 1,
        Alpha = 2,
        Beta = 3,
        Gamma = 4,
    }
}
