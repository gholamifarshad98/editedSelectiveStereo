////////////////////////////////////////////////////////////////////
/// At first you must set stack commit and stack resservd to 78125000
////////////////////////////////////////////////////////////////////
#pragma once
#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<math.h>
#include<vector>
#include<memory>
#include <chrono> 
#include<string> 
#include<math.h>
#include<fstream>
#include<cuda.h>
#include <device_functions.h>
#include <time.h>
#include<stdio.h>
#include<iostream> 
#include<algorithm>



using namespace std;
using namespace cv;


int numOfColumnsResized;
int numOfRowsResized = 0;
int kernelSize = 16;
int maxDisparity = 30;
int selectedDisparity =5;


__global__ void IDAS_Stereo_selective( int MxDisparity, int nC , int nSelected, uchar* leftIm, uchar* rightIm, int* resultIm)
{
	//thread_group my_block = this_thread_block();
	//thread_block block = this_thread_block();
	__shared__   int costs;
	__shared__   int dif[16 * 16];
	int kSize = 16;
	

	int rightPixelIndexU;
	int rightPixelIndexV;

	int leftPixelIndexU;
	int leftPixelIndexV;

	int leftPixelIndex;
	int rightPixelIndex;

	rightPixelIndexU = blockIdx.x + threadIdx.x + MxDisparity + 1;
	rightPixelIndexV = blockIdx.y + threadIdx.y;
	leftPixelIndexU = rightPixelIndexU + (blockIdx.z - 1) + nSelected;
	leftPixelIndexV = rightPixelIndexV;
	leftPixelIndex = leftPixelIndexV * nC + leftPixelIndexU;
	rightPixelIndex = rightPixelIndexV * nC + rightPixelIndexU;
	dif[threadIdx.x + threadIdx.y * kSize] = abs(leftIm[leftPixelIndex] - rightIm[rightPixelIndex]);
	__syncthreads();

	
	if (threadIdx.x == 0 & threadIdx.y == 0) {
		costs = 0;
		for (int i = 0; i < blockDim.y;i++) {
			for(int j=0;j< blockDim.x;j++)
			costs = costs + dif[i + j* kSize];
		}
		//printf("%d \n", costs);
	}

	__syncthreads();

			
	resultIm[((blockIdx.y + int(kSize / 2))* nC + (blockIdx.x + MxDisparity+ int(kSize / 2)+2))*3+ blockIdx.z] = costs;
	
}
void ReadBothImages(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage,int* numOfRows,int* numOfColumns) {

	try {
		//cout << "this is test" << endl;
		*rightImage = cv::imread("2.png", IMREAD_GRAYSCALE);   // Read the right image
																  //rightImage->convertTo(*rightImage, CV_64F);
		*leftImage = cv::imread("1.png", IMREAD_GRAYSCALE);   // Read the left image
		*numOfColumns = ((int)leftImage->cols / 32) * 32;
		*numOfRows = ((int)leftImage->rows / 32) * 32;													 //leftImage->convertTo(*leftImage, CV_64F);
	}
	catch (char* error) {
		cout << "can not load the " << error << " iamge" << endl;
	}

	//cv::resize(*rightImage, *rightImage, cv::Size(), 0.50, 0.50);
	//cv::resize(*leftImage, *leftImage, cv::Size(), 0.50, 0.50);

	//imshow("test", *rightImage);


	//waitKey(5000);
}
int CalcCost(shared_ptr<Mat> leftImage_, shared_ptr<Mat> rightImage_, int row, int column, int kernelSize, int disparity, int NCols) {
	int cost = 0;
	
	for (int u = -int(kernelSize / 2); u <= int(kernelSize / 2); u++) {
		for (int v = -int(kernelSize / 2); v <= int(kernelSize / 2); v++) {
			// for error handeling.
			if (column + u + disparity >= NCols) {
				cout << "*****************************************************" << endl;
			}
			cost = cost + int(pow((leftImage_->at<uchar>(row + v, column + u) - (rightImage_->at<uchar>(row + v, column + u + disparity))), 2));
		}
	}
	return cost;
}
void  SSDstereo(shared_ptr<Mat> leftImage_, shared_ptr<Mat> rightImage_, shared_ptr<Mat> result_temp_, int kernelSize, int maxDisparity, int NRow, int NCols) {
	int tempCost = 0;
	int tempDisparity = 0;

	for (int u = (kernelSize / 2) + 1; u <(NCols - maxDisparity - kernelSize / 2) - 1; u++) {
		for (int v = (kernelSize / 2) + 1; v <NRow - (kernelSize / 2); v++) {
			double cost = 10000000;
			tempCost = 0;
			tempDisparity = 0;
			for (int i = 0; i < maxDisparity; i++) {
				tempCost = CalcCost(leftImage_, rightImage_, v, u, kernelSize, i, NCols);
				if (tempCost < cost) {
					cost = tempCost;
					tempDisparity = i;
				}
			}
			tempDisparity = tempDisparity * 255 / maxDisparity;
			result_temp_->at<uchar>(v, u) = tempDisparity;
			//std::cout << " tempDisparity for ("<< u<<","<<v<<") is "  << tempDisparity << std::endl;
		}
	}
	//std::cout << "debug" << std::endl;
	//cv::imshow("stereoOutput", *result_temp);
	//cv::waitKey(100);
}

int main(void)
{
	//=======================================================================================================================================
	//Memeory Alocation
	//=======================================================================================================================================
	chrono::high_resolution_clock::time_point startTimeReadImage;
	chrono::high_resolution_clock::time_point stopTimeReadImage;
	std::chrono::duration<double, std::milli> durationReadImage;

	chrono::high_resolution_clock::time_point startConvertTo1D;
	chrono::high_resolution_clock::time_point stopConvertTo1D;
	std::chrono::duration<double, std::milli> durationConvertTo1D;

	chrono::high_resolution_clock::time_point startCudaMemcpyInput;
	chrono::high_resolution_clock::time_point stopCudaMemcpyInput;
	std::chrono::duration<double, std::milli> durationCudaMemcpyInput;

	chrono::high_resolution_clock::time_point startCudaCalc;
	chrono::high_resolution_clock::time_point stopCudaCalc;
	std::chrono::duration<double, std::milli> durationCudaCalc;

	chrono::high_resolution_clock::time_point startCudaMemcpyResult;
	chrono::high_resolution_clock::time_point stopCudaMemcpyResult;
	std::chrono::duration<double, std::milli> durationCudaMemcpyResult;

	chrono::high_resolution_clock::time_point startInferenceResult;
	chrono::high_resolution_clock::time_point stopInferenceResult;
	std::chrono::duration<double, std::milli> durationInferenceResult;
	std::chrono::duration<double, std::milli> totalDuraation;



	shared_ptr<Mat> rightImage = make_shared<Mat>();
	shared_ptr<Mat> leftImage = make_shared<Mat>();

	shared_ptr<Mat> rightImageResized = make_shared<Mat>();
	shared_ptr<Mat> leftImageResized = make_shared<Mat>();

	shared_ptr<Mat>  stereoResut = make_shared<Mat>();
	shared_ptr<Mat>  stereoResutResized = make_shared<Mat>();

	int numOfRows;
	int numOfColumns;
	
	//Object for repoting results in a text file.
	ofstream repotringResult;


	//Varaible for convert 2D images to 1D array of images.
	uchar** imArray2DL;
	uchar** imArray2DR;
	int** imArrary2DR_result;
	uchar* imArrary1DL;
	uchar* imArrary1DR;
	int* imArrary1DR_result;
	
	//Varaible for inference the results;
	int firstCost;
	int secondCost;
	int thirdCost;

	//Pointers for Memeory Alocation on GPU.
	uchar* imArray1DL_d;
	uchar* imArray1DR_d;
	int* imArray1DResult_d;




	//=======================================================================================================================================
	//Read Image
	//=======================================================================================================================================
	startTimeReadImage  = chrono::high_resolution_clock::now();
	ReadBothImages(leftImage, rightImage,&numOfRows,&numOfColumns);
	stopTimeReadImage = chrono::high_resolution_clock::now();



	
	

	//=======================================================================================================================================
	//Dynamic Memeory Alocation 
	//=======================================================================================================================================
	stereoResut = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC1);
	imArray2DL= new uchar* [numOfRows];
	imArray2DR = new uchar*[numOfRows];
	imArrary2DR_result= new int*[numOfRows];
	for (int i = 0; i < numOfRows; i++) {
		imArray2DL[i] = new uchar[numOfColumns];
		imArray2DR[i] = new uchar[numOfColumns];
		imArrary2DR_result[i] = new int[numOfColumns*3];
	}
	imArrary1DL = new uchar[numOfColumns * numOfRows];
	imArrary1DR = new uchar[numOfColumns * numOfRows];
	imArrary1DR_result = new int[numOfColumns * numOfRows * 3];
	
	
	cudaMalloc((void**)&imArray1DL_d, numOfColumns * numOfRows * sizeof(uchar));
	cudaMalloc((void**)&imArray1DR_d, numOfColumns * numOfRows * sizeof(uchar));
	cudaMalloc((void**)&imArray1DResult_d, numOfColumns * numOfRows * 3 * sizeof(int));

	// Set grid and bolck size.
	dim3 blocks3D(16, 16, 1);
	dim3 grid2D(numOfColumns - 2 * (maxDisparity + 1) - (kernelSize - 1), numOfRows - kernelSize - 1, 3);





	//=======================================================================================================================================
	//Convert 2D image To 1D  array.
	//=======================================================================================================================================
	startConvertTo1D = chrono::high_resolution_clock::now();
	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			imArray2DL[j][i] = leftImage->at<uchar>(j, i);
			imArray2DR[j][i] = rightImage->at<uchar>(j, i);
		}
	}
	
	for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArrary1DL[i] = imArray2DL[int(i / numOfColumns)][i%numOfColumns];
		imArrary1DR[i] = imArray2DR[int(i / numOfColumns)][i%numOfColumns];
		for(int k=0;k<3;k++){
			imArrary1DR_result[i + i*k] = 0;
		}
	}
	stopConvertTo1D = chrono::high_resolution_clock::now();



	

	//=======================================================================================================================================
	//Copy 1D images to GPU.
	//=======================================================================================================================================
	startCudaMemcpyInput = chrono::high_resolution_clock::now();
	cudaMemcpy(imArray1DL_d, imArrary1DL, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(imArray1DR_d, imArrary1DR, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyHostToDevice);
	stopCudaMemcpyInput = chrono::high_resolution_clock::now();
	



	//=======================================================================================================================================
	//Call kernel to run on GPU.
	//=======================================================================================================================================
	startCudaCalc = chrono::high_resolution_clock::now();
	IDAS_Stereo_selective <<<grid2D, blocks3D >>>(maxDisparity, numOfColumns, selectedDisparity, imArray1DL_d, imArray1DR_d, imArray1DResult_d);
	stopCudaCalc = chrono::high_resolution_clock::now();
	
	


	//=======================================================================================================================================
	//Copy 1D result from GPU.
	//=======================================================================================================================================
	startCudaMemcpyResult = chrono::high_resolution_clock::now();
	cudaMemcpy(imArrary1DR_result, imArray1DResult_d, numOfColumns*numOfRows *3* sizeof(int), cudaMemcpyDeviceToHost);
	stopCudaMemcpyResult = chrono::high_resolution_clock::now();





	
	//=======================================================================================================================================
	//Inference Results.
	//=======================================================================================================================================
	startInferenceResult = chrono::high_resolution_clock::now();
	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			firstCost = imArrary1DR_result[(j * numOfColumns + i)*3];
			secondCost= imArrary1DR_result[(j * numOfColumns + i)*3+1];
			thirdCost= imArrary1DR_result[(j * numOfColumns + i )* 3+2];
			if(secondCost<firstCost& secondCost<thirdCost)
				leftImage->at<uchar>(j, i)=(uchar)255 ;
		}
	}
	stopInferenceResult = chrono::high_resolution_clock::now();

	



	//=======================================================================================================================================
	//Memeory De-alocation.
	//=======================================================================================================================================
	
	cudaFree(imArray1DL_d);
	cudaFree(imArray1DR_d);
	cudaFree(imArray1DResult_d);
	delete imArray2DL;
	delete imArray2DR;
	delete imArrary2DR_result;
	delete imArrary1DL;
	delete imArrary1DR;
	delete imArrary1DR_result;

	





	//=======================================================================================================================================
	//Reporting the results.
	//=======================================================================================================================================
	durationReadImage = stopTimeReadImage - startTimeReadImage;
	durationConvertTo1D= stopConvertTo1D- startConvertTo1D;
	durationCudaMemcpyInput= stopCudaMemcpyInput- startCudaMemcpyInput;
	durationCudaCalc = stopCudaCalc - startCudaCalc;
	durationCudaMemcpyResult = stopCudaMemcpyResult - startCudaMemcpyResult;
	durationInferenceResult = stopInferenceResult - startInferenceResult;
	totalDuraation = durationReadImage + durationConvertTo1D + durationCudaMemcpyInput +
		durationCudaCalc + durationCudaMemcpyResult+ durationInferenceResult;

	string durationReadImage_s = to_string(durationReadImage.count());
	string durationConvertTo1D_s = to_string(durationConvertTo1D.count());
	string durationCudaMemcpyInput_s = to_string(durationCudaMemcpyInput.count());
	string durationCudaCalc_s = to_string(durationCudaCalc.count());
	string durationCudaMemcpyResult_s = to_string(durationCudaMemcpyResult.count());
	string durationInferenceResult_s = to_string(durationInferenceResult.count());
	string totalDuraation_s = to_string(totalDuraation.count());

	repotringResult.open("results.txt");
	repotringResult << "durationReadImage = " << durationReadImage_s << endl;
	repotringResult << "durationConvertTo1D = " << durationConvertTo1D_s << endl;
	repotringResult << "durationCudaMemcpyInput = " << durationCudaMemcpyInput_s << endl;
	repotringResult << "durationCudaCalc = " << durationCudaCalc_s << endl;
	repotringResult << "durationCudaMemcpyResult = " << durationCudaMemcpyResult_s << endl;
	repotringResult << "durationInferenceResult = " << durationInferenceResult_s << endl;
	repotringResult << "totalDuraation = " << totalDuraation_s << endl;
	repotringResult.close();




	imshow(" Left after calaculation !!!", *leftImage);
	imwrite("result.png", *leftImage);
	imshow("Right image !!!   .....", *rightImage);
	waitKey(1000);
	printf("\n \n \n  \t \t \t :)  ");
	char str[80];
	scanf("%79s", str);
}




//command for syncronization of thread.
//cudaDeviceSynchronize();

//copy data to 2d reasult image.
	/*for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArrary2DR_result[int(i / numOfColumns)][i%numOfColumns] = imArrary1DR_result[i];
	}*/