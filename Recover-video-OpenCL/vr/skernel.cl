__kernel void recover_video( __global unsigned char* R,
                             __global unsigned char* G,
                             __global unsigned char* B,
                             __global float * ans,
                             int N , int H, int W)
{
    int i = get_global_id(0);
	if(i>=N*(N-1)/2)
		return;
    int sub = N-1;
    int cnt =0;
    int n1,n2;
    while(1)
    {
        if(i<sub || sub==0)
           break;
        i -= sub;
		sub--;
        cnt++;
    }
    n1 = cnt;
    n2 = cnt+1+i;
    
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
//    ans[n1 * N + n2] = get_global_id(0);
//    ans[n2 * N + n1] = get_global_id(0);
}
