__kernel void recover_video( __global unsigned char* R,
                             __global unsigned char* G,
                             __global unsigned char* B,
                             __global float * ans,
                             int N , int H, int W)
{
    int n1 = get_global_id(1); //N
    int n2 = get_global_id(0); //N

    if(n1 >= n2 || n1 >= N || n2 >= N)
        return;
    
    int h,w;
    int d;
    float dImg = 0;
    for(h=0;h<H;h++)
    {
        for(w=0;w<W;w++)
        {
            int dPixel =0;  
            d = (int)R[(n1 * H + h) * W + w] - (int)R[(n2 * H + h) * W + w];
            dPixel += d * d;
            d = (int)G[(n1 * H + h) * W + w] - (int)G[(n2 * H + h) * W + w];
            dPixel += d * d;
            d = (int)B[(n1 * H + h) * W + w] - (int)B[(n2 * H + h) * W + w];
            dPixel += d * d;
            dImg += sqrt((float)dPixel); //calculate diff
        }
    }
    ans[n1 * N + n2] = dImg;
    ans[n2 * N + n1] = dImg;
}
