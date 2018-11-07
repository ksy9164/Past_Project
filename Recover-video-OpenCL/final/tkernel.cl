__kernel void recover_R( __global unsigned char *R,
                         __global long *rcvR,
                         int N, int H, int W)
{
    int frame1 = get_global_id(1);
    int frame2 = get_global_id(0);

    if(frame1 >= frame2 || N <= frame1 || N <= frame2)
        return;

    int height, width;
    int d;
    long dImg = 0;

    for(height = 0; height < H; ++height)
    {
        for(width = 0; width < W; ++width)
        {
            d = (int)R[(frame1 * H + height) * W + width] - (int)R[(frame2 * H + height) * W + width];
            dImg += (long)(d * d);
        }
    }

    rcvR[frame1 * N + frame2] = dImg;
    rcvR[frame2 * N + frame1] = dImg;
}

__kernel void recover_G( __global unsigned char *G,
                         __global long *rcvG,
                         int N, int H, int W)
{
    int frame1 = get_global_id(1);
    int frame2 = get_global_id(0);

    if(frame1 >= frame2 || N <= frame1 || N <= frame2)
        return;

    int height, width;
    int d;
    long dImg = 0;

    for(height = 0; height < H; ++height)
    {
        for(width = 0; width < W; ++width)
        {
            d = (int)G[(frame1 * H + height) * W + width] - (int)G[(frame2 * H + height) * W + width];
            dImg += (long)(d * d);
        }
    }

    rcvG[frame1 * N + frame2] = dImg;
    rcvG[frame2 * N + frame1] = dImg;
}

__kernel void recover_B( __global unsigned char *B,
                         __global long *rcvB,
                         int N, int H, int W)
{
    int frame1 = get_global_id(1);
    int frame2 = get_global_id(0);

    if(frame1 >= frame2 || N <= frame1 || N <= frame2)
        return;

    int height, width;
    int d;
    long dImg = 0;

    for(height = 0; height < H; ++height)
    {
        for(width = 0; width < W; ++width)
        {
            d = (int)B[(frame1 * H + height) * W + width] - (int)B[(frame2 * H + height) * W + width];
            dImg += (long)(d * d);
        }
    }

    rcvB[frame1 * N + frame2] = dImg;
    rcvB[frame2 * N + frame1] = dImg;
}
