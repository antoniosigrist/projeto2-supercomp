#include <iostream>
#include <fstream>
using namespace std;
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include <chrono>
using namespace std::chrono;

#define RND (curand_uniform(&local_rand_state))
#define seed 1000

 __device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); 
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, 0, rand_state); //cria uma seed na thread 0 que vai garantir mesma seed para todas as threads
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) 
        return;
    int pixel_index = j*max_x + i;
    
    curand_init(seed, pixel_index, 0, &rand_state[pixel_index]); //repassa o mesmo rand_state para os threads dos blocos utilizados na compilacao para garantir consistencia. 
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) 
        return; // garante que nao vai rodar alem dos tamanho definido no kernel
    int pixel_index = j*max_x + i; // calcula a posicao do pixel no kernel
    curandState local_rand_state = rand_state[pixel_index]; 
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);

        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col; //coloca o resultado em fb para ser acessado do host ao final do código
}


__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2, new lambertian(vec3(curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state), RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2, new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(25,12,13);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) { //deleta espaco alocado para cada esfera
    for(int i=0; i < 22*22+1+3; i++) { 
        delete ((sphere *)d_list[i])->mat_ptr; 
        delete d_list[i];
    }
    delete *d_world; 
    delete *d_camera;
}

int main() {

    ofstream myfile;
    myfile.open ("tempo.txt");

    int num_testes = 30;

    int prop;
    
    for(int k = 1;k<num_testes;k++) {

    prop = k;

    int nx = (int) 1200/prop;
    int ny = (int) 800/prop;
    int ns = 10;
    int tx = 8;
    int ty = 8;


    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3); //aloca tamanho do vetor fb para cada pixel caber um vec3


    vec3 *fb;
    cudaMallocManaged((void **)&fb, fb_size); //aloca lista do tamanho do numero de pixels da imagem. Cudamallocmanage "copia" o mesmo endereco de memoria para CPU e GPU.

  
    curandState *d_rand_state;
    cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState));
    curandState *d_rand_state2;
    cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState));

    rand_init<<<1,1>>>(d_rand_state2); // inicializa kernel que cria seed no bloco 0 thread 0

    //cudaDeviceSynchronize();  - Utilizado apenas para debug, não é necessário dado que threads são assincrona entre si, mas sequenciais entre elas

    hitable **d_list;
    int num_hitables = 489; //tem que ser maior ou igual a 488 que é o numero de bolinhas criadas :)
    cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)); //aloca hitables
    hitable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hitable *));
    camera **d_camera;
    cudaMalloc((void **)&d_camera, sizeof(camera *));

    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2); //cria mundo randomico utilizando o estado incial criado por rand_init no bloco 0 thread 0


    clock_t start, stop;
    start = clock();

    dim3 blocks(nx/tx+1,ny/ty+1); //define o numero de blocos (tx e ty são multiplos de 8 já que a arquitetura de 8x8 threads, garantindo que cada bloco faca um numero pareceido de processamento)
    dim3 threads(tx,ty);

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state); //cria o kernel de tamanho block x threads. 


    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state); // renderiza a imagem no tamanho do bloco e threads estabelicidos, garantindo o mesmo cenário para todas as threads (maior parte do processamento está aqui)


    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;


    myfile << "Tamanho da Imagens x Tempo de Execução: ";
    myfile << "\n";
    myfile << "Tamanho da Imagem: "<< nx <<" x " << ny << " - Tempo de Execução: " << timer_seconds << "," << "\n"; //escreve tempo de execucao e tamanho da imagem da imagem rodada.

    // Como estamos realizando diversos testes de tamanhos de imagem diferente, desejamos que apenas uma imagem seja criada para podermos analisar a qualidade
    if(k==2){ //devolve apenas os pixels do tamanho de prop==2

        cudaDeviceSynchronize(); //garante que processamento já acabou para acessar dados de fb

        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j*nx + i;
                int ir = int(255.99*fb[pixel_index].r()); //pega cor vermelha de fb
                int ig = int(255.99*fb[pixel_index].g()); // pega cor verde de fb
                int ib = int(255.99*fb[pixel_index].b());//pega cor azul de fb
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }
    }

    // limpando a memoria alocada

    free_world<<<1,1>>>(d_list,d_world,d_camera);

    cudaFree(d_list);
    cudaFree(d_rand_state);
    cudaFree(fb);
    cudaFree(d_camera);
    cudaFree(d_world);
    

    cudaDeviceReset();

    }

    myfile.close(); //fecha arquivo de escrita
}









