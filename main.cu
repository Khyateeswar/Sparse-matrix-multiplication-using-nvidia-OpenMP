#include <bits/stdc++.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <omp.h>

using namespace std;





__global__ void matmul(int *a, int *b, int *c,int m,int u,int n,int *p1,int* p2,int* p3) {
  //p1 has the index to access i,j block from a
  //p2 has the index to access i,j block from b
  //p3 contains information about non zero blocks
  //c is the final matrix
  int i = blockIdx.x/u;
  int j = blockIdx.x%u;
  int l = threadIdx.x;
  int s = u/(blockDim.x)+1;
  for(int k=s*l;k<s*l+s;k++){

    if(i<u && j<u && k<u){
    if(p1[i*u+k]>=0 && p2[k*u+j]>=0){
      atomicOr(p3+i*u+j,1);
      int ik = p1[i*u+k];
      int kj = p2[k*u+j];
      for(int p=0;p<m;p++){
        for(int q=0;q<m;q++){
          for(int z=0;z<m;z++){
            atomicAdd(c+(i*m+p)*n+j*m+q,b[ kj*m*m+z*m+q]*a[ ik*m*m+p*m+z ]);

          }
        }
      }
    }
  }


  }

  
}

string conv4(int value)
{
    int leftmost_byte;
    int left_middle_byte;
    int right_middle_byte;
    int rightmost_byte;
    
    leftmost_byte = (value & 0x000000FF) >> 0;
    left_middle_byte = (value & 0x0000FF00) >> 8;
    right_middle_byte = (value & 0x00FF0000) >> 16;
    rightmost_byte = (value & 0xFF000000) >> 24;
    string s = "";
    char c=*reinterpret_cast<char *>(&leftmost_byte );
    s=s+c;
    c=*reinterpret_cast<char *>(&left_middle_byte );
    s=s+c;
    c=*reinterpret_cast<char *>(&right_middle_byte );
    s=s+c;
    c=*reinterpret_cast<char *>(&rightmost_byte );
    s=s+c;
    return s;
}



int main(int argc, char** argv) {

  int max = omp_get_max_threads();
cout<<max<<endl;

  if(argc!=4){
        cout<<"Check command line arguments"<<endl;
        return 0;
  }
  char* input_file1 = argv[1];
  char* input_file2 = argv[2];
  char* output_file = argv[3];

  clock_t start,end;
    double timetaken;

    start = clock();


    int n,m,k;

    ifstream ifs;
    ifs.open (input_file1, ios::in | ios::binary );
    ifs.seekg (0, ios::end);
    int length = ifs.tellg();
    ifs.seekg (0, ios::beg);
    char* buf = new char[length];
    ifs.read (buf,length);
    ifs.close();

    n = *reinterpret_cast<int *>( buf );
    m = *reinterpret_cast<int *>( buf+4 );
    k = *reinterpret_cast<int *>( buf+8 );
    int k1=k;
    int u = n/m;

    cout<<n<<" "<<m<<" "<<k<<" "<<endl;

    int mat1[k*m*m];
    // #pragma omp parallel for schedule(static)
    for(int i=0;i<k*m*m;i++){
        mat1[i]=0;
    }
    int p1[u*u];
    for(int i=0;i<u*u;i++){
        p1[i]=-1;
    }

    
    for(int i=0;i<k;i++){
        int ind = *reinterpret_cast<int *>( buf+(8+m*m*2)*i+12 );
        int jnd = *reinterpret_cast<int *>( buf+(8+m*m*2)*i+4+12);
        p1[ind*u+jnd]=i;
        int ofss = i*m*m;
        int h = (8+m*m*2)*i+8+12;
        for(int j=0;j<m;j++){
            for(int q=0;q<m;q++){
                int val = *reinterpret_cast<uint16_t *>( buf+h);
                mat1[ ofss+j*m+q ] = val;
                h=h+2;
            }
        }
    }

    delete [] buf;

    // for(int i=0;i<k*m*m;i++){
    //   cout<<mat1[i]<<" ";
    // }
    // cout<<'\n';


    // for(int i=0;i<u*u;i++){
    //   cout<<p1[i]<<" ";
    // }
    // cout<<'\n';

    ifs.open (input_file2, ios::in | ios::binary );
    ifs.seekg (0, ios::end);
    length = ifs.tellg();
    ifs.seekg (0, ios::beg);
    char* buf1 = new char[length];
    ifs.read (buf1,length);
    ifs.close();

    n = *reinterpret_cast<int *>( buf1 );
    m = *reinterpret_cast<int *>( buf1+4 );
    k = *reinterpret_cast<int *>( buf1+8 );

    cout<<n<<" "<<m<<" "<<k<<" "<<endl;

    int mat2[k*m*m];
    // #pragma omp parallel for schedule(static)
    for(int i=0;i<k*m*m;i++){
        mat2[i]=0;
    }

    int p2[u*u];
    for(int i=0;i<u*u;i++){
        p2[i]=-1;
    }
    int p3[u*u];
    for(int i=0;i<u*u;i++){
        p3[i]=0;
    }
    for(int i=0;i<k;i++){
        int ind = *reinterpret_cast<int *>( buf1+(8+m*m*2)*i+12 );
        int jnd = *reinterpret_cast<int *>( buf1+(8+m*m*2)*i+4+12);
        p2[ind*u+jnd]=i;
        int ofss = i*m*m;
        int h = (8+m*m*2)*i+8+12;
        for(int j=0;j<m;j++){
            for(int q=0;q<m;q++){
                int val = *reinterpret_cast<uint16_t *>( buf1+h);
                mat2[  ofss+j*m+q ] = val;
                h=h+2;;
            }
        }
    }
    delete [] buf1;

    end = clock();
    timetaken = (end - start) / (double)CLOCKS_PER_SEC;
    cout << "Time taken by taking input: " << fixed << timetaken << "s" << endl;

    // for(int i=0;i<k*m*m;i++){
    //   cout<<mat2[i]<<" ";
    // }
    // cout<<'\n';

    // for(int i=0;i<u*u;i++){
    //   cout<<p2[i]<<" ";
    // }
    // cout<<'\n';

    int mat3[n*n];






// host copies of variables a, b & c
  int *m1, *m2, *m3;
  int *c1, *c2, *c3;
// device copies of variables a, b & c
  int size2 = (k*m*m)*sizeof(int);
  int size1 = (k1*m*m)*sizeof(int);
  int usize = (u*u)*sizeof(int);

  start = clock();

// Allocate space for device copies of a, b, c
  cudaMalloc((void **)&m1, size1);
  cudaMalloc((void **)&m2, size2);
  cudaMalloc((void **)&m3, (n*n)*sizeof(int));
  cudaMalloc((void **)&c1, usize);
  cudaMalloc((void **)&c2, usize);
  cudaMalloc((void **)&c3, usize);
// Setup input values  
// c = 0;
// a = 3;
// b = 5;
// Copy inputs to device
  cudaDeviceSynchronize();
  cudaMemcpy(m1, &mat1[0], size1, cudaMemcpyHostToDevice);
  cudaMemcpy(c1, &p1[0], usize, cudaMemcpyHostToDevice);
//cout<<*(d_a+0)<<" "<<*(d_a+1)<<" "<<*(d_a+2)<<endl;
  cudaMemcpy(m2, &mat2[0], size2, cudaMemcpyHostToDevice);
  cudaMemcpy(c2, &p2[0], usize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
//cout<<*(d_b+0)<<" "<<*(d_b+1)<<" "<<*(d_b+2)<<endl;
// Launch add() kernel on GPU
  matmul<<<u*u,1000>>>(m1,m2,m3,m,u,n,c1,c2,c3);
//cout<<*(d_c+0)<<" "<<*(d_c+1)<<" "<<*(d_c+2)<<endl;
// Copy result back to host
  cudaDeviceSynchronize();
  cudaMemcpy(&mat3[0], m3,(n*n)*sizeof(int) , cudaMemcpyDeviceToHost);
  cudaMemcpy(&p3[0], c3, usize, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

//   if(err!=cudaSuccess) {
//       printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
//   }
// printf("result is %d\n",c);
// Cleanup
  cudaFree(m1);
  cudaFree(m2);
  cudaFree(m3);
  cudaFree(c1);
  cudaFree(c2);
  cudaFree(c3);

  end = clock();
  timetaken = (end - start) / (double)CLOCKS_PER_SEC;
  cout << "Time taken by matrix mul: " << fixed << timetaken << "s" << endl;

  // cout<<" cuda file"<<endl;

  // for(int i=0;i<n;i++){
  //   for(int j=0;j<n;j++){
  //     cout<<mat3[i*n+j]<<" ";
  //   }
  //   cout<<'\n';
  // }

  // for(int i=0;i<u*u;i++){
  //     cout<<p3[i]<<" ";
  //   }
  //   cout<<'\n';



//output the file


  string fin = "";
    fin=fin+conv4(n);
    fin=fin+conv4(m);
    fin=fin+"    ";
    int count = 0;
    #pragma omp parallel for collapse(2)
    for(int i=0;i<u;i++){
        for(int j=0;j<u;j++){
            //printf("i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
            if( p3[i*u+j]>0 ){
            string s = "";
            s=s+conv4(i);
            s=s+conv4(j);
            //cout<<i<<" "<<j<<endl;

            for(int a=0;a<m;a++){
                for(int b=0;b<m;b++){
                    s+=conv4(mat3[(i*m+a)*n+j*m+b]);
                    //cout<<mat3[(i*m+a)*n+j*m+b]<<" ";
                }
                //cout<<'\n';
            }
            #pragma omp critical
            {
            fin =fin+s;
            count++;
            //cout<<count<<endl;
            }
            }
        }
    }

    string cs = conv4(count);
    fin[8]=cs[0];
    fin[9]=cs[1];
    fin[10]=cs[2];
    fin[11]=cs[3];
    ofstream of(output_file,ios::out | ios::binary);
    of.write((char*)&fin[0],fin.size());
    of.close();


  return 0;
}
