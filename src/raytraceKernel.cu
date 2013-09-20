// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include <time.h>  
#include <windows.h>

#define LIGHT_NUM 15
#define ANTI_NUM 2

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	system("pause");
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__host__ __device__ int generateRandomNumber(glm::vec2 resolution, float time, int x, int y)
{
	int index = x + (y * resolution.x);
   
	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return (int)(u01(rng)) * 10;
}


//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  
	//printf("%f  ", x);

	glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);
	glm::vec3 M = eye + view;

	float absC = glm::length(view);
	float absA = glm::length(A);
	float absB = glm::length(B);

	glm::vec3 H = (A * absC * (float)tan(fov.x*(PI/180))) / absA;
	glm::vec3 V = (B * absC * (float)tan(fov.y*(PI/180))) / absB;

	glm::vec3 P = M + (2 * (x / (float)(resolution.x - 1)) - 1) * H + (1 - 2 * (y / (float)(resolution.y - 1))) * V;
	glm::vec3 D = (P - eye) / glm::length(P - eye);

	ray r;
	r.origin = eye;
	r.direction = D;	
	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;

      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}


__host__ __device__  bool ShadowRayUnblock(staticGeom* geoms, int numberofGeoms, glm::vec3 intersectionPoint, int lightIndex, int geomIndex, glm::vec3 normal, glm::vec3 lightPos)
{	
	ray r;	
	r.direction = glm::normalize(lightPos - intersectionPoint);
	r.origin = intersectionPoint + .1f * r.direction;

	//surface parallel to the light
	if(glm::dot(r.direction, normal) < 1e-5 && glm::dot(r.direction, normal) > -1e-5)
		return false;

	glm::vec3 tempInterPoint, tempNormal;	
	float t = FLT_MAX;
	int intersectIndex = -1;
	for(int i = 0; i < numberofGeoms; ++i)
	{		
		float temp;
		if(geoms[i].type == CUBE)
			temp = boxIntersectionTest(geoms[i], r, tempInterPoint, tempNormal);
		else
			temp = sphereIntersectionTest(geoms[i], r, tempInterPoint, tempNormal);

		if(temp < t && temp != -1.0f)
		{
			t = temp;
			intersectIndex = i;
		}
	}

	if(intersectIndex == lightIndex)
		return true;
	else
		return false;
}

__host__ __device__ bool checkIntersect(staticGeom* geoms, int numberOfGeoms, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, int& geomIndex)
{
	float t = FLT_MAX;
	//int geomIndex = 0;
	for(int i = numberOfGeoms - 1; i >= 0; --i)
	{	
		float temp;
		//if(geoms[i].materialid != 7 || geoms[i].materialid != 8)
		if(geoms[i].type == SPHERE)
			temp = sphereIntersectionTest(geoms[i], r, intersectionPoint, normal);
		else if(geoms[i].type == CUBE)
			temp = boxIntersectionTest(geoms[i], r, intersectionPoint, normal);

		if(temp != -1.0f && temp < t)
		{
			t = temp;
			geomIndex = i;		
		}
	}	

	if(t != FLT_MAX){
		//get the intersection point and normal
		if(geoms[geomIndex].type == SPHERE)
			sphereIntersectionTest(geoms[geomIndex], r, intersectionPoint, normal);
		else
			boxIntersectionTest(geoms[geomIndex], r, intersectionPoint, normal);
		return true;
	}
	else
	{
		return false;
	}
}

__host__ __device__ void TraceRay(ray r, int rayDepth, staticGeom* geoms, int numberOfGeoms, material* materials, glm::vec3& color, 
								  glm::vec3 eyePosition, glm::vec3* lightPos, int lightIndex, float time, int randomLightPos)
{
	if(rayDepth > 3)
	{
		color = glm::vec3(0,0,0);
		return;
	}

	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	int geomIndex = 0;

	bool isIntersect = checkIntersect(geoms, numberOfGeoms, r, intersectionPoint, normal, geomIndex);		
	
	if(!isIntersect)//no intersection
	{
		color = glm::vec3(0,0,0);
		return;
	}
	else 
	{				
		material currMaterial =  materials[geoms[geomIndex].materialid];

		if(geomIndex == lightIndex)
		{
			color = currMaterial.color * currMaterial.emittance;
			return;
		}

		glm::vec3 spec, refr;
		glm::vec3 reflectedColor, refractedColor;

		if(currMaterial.hasReflective > 0.0f)//Reflective
		{			
			glm::vec3 reflectionDirection = calculateReflectionDirection(normal, r.direction);
			ray newRay;
			newRay.origin = intersectionPoint + 0.001f * reflectionDirection;
			newRay.direction = reflectionDirection;	
			
			TraceRay(newRay, rayDepth + 1, geoms, numberOfGeoms, materials, reflectedColor, eyePosition, lightPos, lightIndex, time, randomLightPos);			

			float reflective = currMaterial.hasReflective;
			spec = reflective * reflectedColor;
		}
			

		if(currMaterial.hasRefractive > 0.0f)//Refractive
		{
			float refractive = currMaterial.indexOfRefraction;

			float inOrOut = glm::dot(r.direction, normal);
			glm::vec3 refractedDirection;
			
			if(inOrOut < 0)
			{
				refractedDirection = calculateTransmissionDirection(normal, r.direction, 1.0, refractive); 
			}
			else
			{
				refractedDirection = calculateTransmissionDirection(-normal, r.direction, refractive, 1.0); 
			}			
			ray newRay;
			newRay.origin = intersectionPoint + 0.001f * refractedDirection;
			newRay.direction = refractedDirection;

			TraceRay(newRay, rayDepth + 1, geoms, numberOfGeoms, materials, refractedColor, eyePosition, lightPos, lightIndex, time, randomLightPos);
			refr = currMaterial.hasRefractive * refractedColor;
		}
		
		
		for(int i = 0; i < LIGHT_NUM; i++)
		{
			glm::vec3 currlightPos = lightPos[i];
			color += .2f * currMaterial.color + 0.7f * refr;
			
			if(ShadowRayUnblock(geoms, numberOfGeoms, intersectionPoint, lightIndex, geomIndex, normal, currlightPos))
			{
				float diffuseTerm;			
				diffuseTerm = glm::dot(glm::normalize(currlightPos - intersectionPoint), normal);
				diffuseTerm = max(diffuseTerm, 0.0f);

				float specTerm = glm::dot(calculateReflectionDirection(normal, (intersectionPoint - currlightPos)), glm::normalize(eyePosition - intersectionPoint));
				specTerm = max(specTerm, 0.0f);

				float specNum;
				if(currMaterial.specularExponent > 0)
					specNum = 0.7f * pow(specTerm, currMaterial.specularExponent);
				else
					specNum = 0.0f;

				color += (materials[geoms[lightIndex].materialid].color * (0.5f * currMaterial.color * diffuseTerm + specNum)) + 0.2f * spec;
			}
		}
		//float area = geoms[lightIndex].scale.x * geoms[lightIndex].scale.y * geoms[lightIndex].scale.z;

		color /= LIGHT_NUM;				
		return;
	}	
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, staticGeom* geoms,
							int numberOfGeoms, material* materials, int numberOfMaterials, glm::vec3* lightPos, int lightIndex)
{

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y))
	{
		colors[index] = glm::vec3(0,0,0);
		int ns = ANTI_NUM;
		glm::vec3 color;
		for(int sx = 0; sx < ns; sx ++)
		{
			for(int sy = 0; sy < ns; sy++)
			{
				//get random number for anti-alaising
				glm::vec3 ran = generateRandomNumberFromThread(cam.resolution, time, x, y);
				ray r = raycastFromCameraKernel(resolution, time, (float)(x + (sx + ran.x)/ns), (float)(y + (sy + ran.y)/ns), cam.position, cam.view, cam.up, cam.fov);
				TraceRay(r, rayDepth, geoms, numberOfGeoms, materials, color, cam.position, lightPos, lightIndex, time, (int)(ran.z * 10));		
				colors[index] += color;
			}
		}	
		colors[index] /= float(ns*ns);
	}
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  SYSTEMTIME st, et;
  GetSystemTime(&st);
  time_t timer1 = time(NULL) * 1000;
  //time(&timer1) * 1000;
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;	
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
 
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  //passing material
  material* cudamaterial = NULL;
  cudaMalloc((void**)&cudamaterial, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterial, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  int lightIndex;
  for(int i = 0; i < numberOfGeoms; i++)
  {
	  if(geomList[i].materialid == 7 || geoms[i].materialid == 8)
	  {
		  lightIndex = i;
		  break;
	  }
  }

  int lightNum = LIGHT_NUM;
  glm::vec3 *lightPos = new glm::vec3[lightNum];
  for(int i = 0; i < lightNum; i++)
  {
	  lightPos[i] = getRandomPointOnCube(geomList[lightIndex], (float)i);
  }

  glm::vec3* cudaLightPos = NULL;
  cudaMalloc((void**)&cudaLightPos, lightNum*sizeof(glm::vec3));
  cudaMemcpy(cudaLightPos, lightPos, lightNum*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  size_t size;
  cudaDeviceSetLimit(cudaLimitStackSize, 10000*sizeof(float));
  cudaDeviceGetLimit(&size, cudaLimitStackSize);

  checkCUDAError("pre-raytraceRay error");

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, 
	  numberOfGeoms, cudamaterial, numberOfMaterials, cudaLightPos, lightIndex);
  
  checkCUDAError("raytraceRay error");
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
 
  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //for(int i = 0; i < (int)renderCam->resolution.x*(int)renderCam->resolution.y; i++)
	//   renderCam->image[i] = renderCam->image[i] / (float)iterations;


  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  time_t timer2 =  time(NULL) * 1000;
  //time(&timer2) * 1000;
  GetSystemTime(&et);

  double seconds;
  seconds = et.wMilliseconds - st.wMilliseconds;//difftime(timer2, timer1);

  printf(" %f \n",seconds);

  checkCUDAError("Kernel failed!");
}
