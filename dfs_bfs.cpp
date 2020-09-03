 
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
 
using namespace std;
using namespace cv; 
 
 
//------------------------------【两步法】----------------------------------------------
// 对二值图像进行连通区域标记,从1开始标号
void  Two_PassNew( const Mat &bwImg, Mat &labImg )
{
	assert( bwImg.type()==CV_8UC1 );
	labImg.create( bwImg.size(), CV_32SC1 );   //bwImg.convertTo( labImg, CV_32SC1 );
	labImg = Scalar(0);
	labImg.setTo( Scalar(1), bwImg );
	assert( labImg.isContinuous() );
	const int Rows = bwImg.rows - 1, Cols = bwImg.cols - 1;
	int label = 1;
	vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);
	//the first pass
	int *data_prev = (int*)labImg.data;   //0-th row : int* data_prev = labImg.ptr<int>(i-1);
	int *data_cur = (int*)( labImg.data + labImg.step ); //1-st row : int* data_cur = labImg.ptr<int>(i);
	for( int i = 1; i < Rows; i++ )
	{
		data_cur++;
		data_prev++;
		for( int j=1; j<Cols; j++, data_cur++, data_prev++ )
		{
			if( *data_cur!=1 )
				continue;
			int left = *(data_cur-1);
			int up = *data_prev;
			int neighborLabels[2];
			int cnt = 0;
			if( left>1 )
				neighborLabels[cnt++] = left;
			if( up > 1)
				neighborLabels[cnt++] = up;
			if( !cnt )
			{
				labelSet.push_back( ++label );
				labelSet[label] = label;
				*data_cur = label;
				continue;
			}
			int smallestLabel = neighborLabels[0];
			if( cnt==2 && neighborLabels[1]<smallestLabel )
				smallestLabel = neighborLabels[1];
			*data_cur = smallestLabel;
			// 保存最小等价表
			for( int k=0; k<cnt; k++ )
			{
				int tempLabel = neighborLabels[k];
				int& oldSmallestLabel = labelSet[tempLabel];  //这里的&不是取地址符号,而是引用符号
				if( oldSmallestLabel > smallestLabel )
				{
					labelSet[oldSmallestLabel] = smallestLabel;
					oldSmallestLabel = smallestLabel;
				}
				else if( oldSmallestLabel<smallestLabel )
					labelSet[smallestLabel] = oldSmallestLabel;
			}
		}
		data_cur++;
		data_prev++;
	}
	//更新等价队列表,将最小标号给重复区域
	for( size_t i = 2; i < labelSet.size(); i++ )
	{
		int curLabel = labelSet[i];
		int prelabel = labelSet[curLabel];
		while( prelabel != curLabel )
		{
			curLabel = prelabel;
			prelabel = labelSet[prelabel];
		}
		labelSet[i] = curLabel;
	}
	//second pass
	data_cur = (int*)labImg.data;
	for( int i = 0; i < Rows; i++ )
	{
		for( int j = 0; j < bwImg.cols-1; j++, data_cur++)
			*data_cur = labelSet[ *data_cur ];
		data_cur++;
	}
}
 
 
 
//-------------------------------------------【种子填充法】---------------------------
void SeedFillNew(const cv::Mat& _binImg, cv::Mat& _lableImg )
{
	// connected component analysis(4-component)
	// use seed filling algorithm
	// 1. begin with a forgeground pixel and push its forground neighbors into a stack;
	// 2. pop the pop pixel on the stack and label it with the same label until the stack is empty
	// 
	//  forground pixel: _binImg(x,y)=1
	//  background pixel: _binImg(x,y) = 0
 
 
	if(_binImg.empty() ||
		_binImg.type()!=CV_8UC1)
	{
		return;
	} 
 
	_lableImg.release();
	_binImg.convertTo(_lableImg,CV_32SC1);
 
	int label = 0; //start by 1
 
	int rows = _binImg.rows;
	int cols = _binImg.cols;
 
	Mat mask(rows, cols, CV_8UC1);
	mask.setTo(0);
	int *lableptr;
	for(int i=0; i < rows; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		uchar *masKptr = mask.ptr<uchar>(i);
		for(int j = 0; j < cols; j++)
		{
			if(data[j] == 255&&mask.at<uchar>(i,j)!=1)
			{
				mask.at<uchar>(i,j)=1;
				std::stack<std::pair<int,int>> neighborPixels;
				neighborPixels.push(std::pair<int,int>(i,j)); // pixel position: <i,j>
				++label; //begin with a new label
				while(!neighborPixels.empty())
				{
					//get the top pixel on the stack and label it with the same label
					std::pair<int,int> curPixel =neighborPixels.top();
					int curY = curPixel.first;
					int curX = curPixel.second;
					_lableImg.at<int>(curY, curX) = label;
 
					//pop the top pixel
					neighborPixels.pop();
 
					//push the 4-neighbors(foreground pixels)
 
					if(curX-1 >= 0)
					{
						if(_lableImg.at<int>(curY,curX-1) == 255&&mask.at<uchar>(curY,curX-1)!=1) //leftpixel
						{
							neighborPixels.push(std::pair<int,int>(curY,curX-1));
							mask.at<uchar>(curY,curX-1)=1;
						}
					}
					if(curX+1 <=cols-1)
					{
						if(_lableImg.at<int>(curY,curX+1) == 255&&mask.at<uchar>(curY,curX+1)!=1)
							// right pixel
						{
							neighborPixels.push(std::pair<int,int>(curY,curX+1));
							mask.at<uchar>(curY,curX+1)=1;
						}
					}
					if(curY-1 >= 0)
					{
						if(_lableImg.at<int>(curY-1,curX) == 255&&mask.at<uchar>(curY-1,curX)!=1)
							// up pixel
						{
							neighborPixels.push(std::pair<int,int>(curY-1, curX));
							mask.at<uchar>(curY-1,curX)=1;
						}  
					}
					if(curY+1 <= rows-1)
					{
						if(_lableImg.at<int>(curY+1,curX) == 255&&mask.at<uchar>(curY+1,curX)!=1)
							//down pixel
						{
							neighborPixels.push(std::pair<int,int>(curY+1,curX));
							mask.at<uchar>(curY+1,curX)=1;
						}
					}
				}
			}
		}
	}
}
 
 
 
//---------------------------------【颜色标记程序】-----------------------------------
//彩色显示
cv::Scalar GetRandomColor()
{
	uchar r = 255 * (rand()/(1.0 + RAND_MAX));
	uchar g = 255 * (rand()/(1.0 + RAND_MAX));
	uchar b = 255 * (rand()/(1.0 + RAND_MAX));
	return cv::Scalar(b,g,r);
}
 
 
void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg) 
{
	int num = 0;
	if (labelImg.empty() ||
		labelImg.type() != CV_32SC1)
	{
		return;
	}
 
	std::map<int, cv::Scalar> colors;
 
	int rows = labelImg.rows;
	int cols = labelImg.cols;
 
	colorLabelImg.release();
	colorLabelImg.create(rows, cols, CV_8UC3);
	colorLabelImg = cv::Scalar::all(0);
 
	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)labelImg.ptr<int>(i);
		uchar* data_dst = colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = GetRandomColor();
					num++;
				}
 
				cv::Scalar color = colors[pixelValue];
				*data_dst++   = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
	printf("color num : %d \n", num );
}
 
//------------------------------------------【测试主程序】--------------------------------
int main()
{
 
	cv::Mat binImage = cv::imread("ltc2.jpg", 0);
	//cv::threshold(binImage, binImage, 50, 1, CV_THRESH_BINARY);
	cv::Mat labelImg;
	double time;
	time= getTickCount();
	//对应四种方法，需要哪一种，则调用哪一种
	//Two_PassOld(binImage, labelImg);
	//Two_PassNew(binImage, labelImg);
	//SeedFillOld(binImage, labelImg);
	//SeedFillNew(binImage, labelImg);
	time = 1000*((double)getTickCount() - time)/getTickFrequency();
    cout<<std::fixed<<time<<"ms"<<endl;
	//彩色显示
	cv::Mat colorLabelImg;
	LabelColor(labelImg, colorLabelImg);
	cv::imshow("colorImg", colorLabelImg);
	//灰度显示
	cv::Mat grayImg;
	labelImg *= 10;
	labelImg.convertTo(grayImg, CV_8UC1);
	cv::imshow("labelImg", grayImg);
	double minval, maxval;
	minMaxLoc(labelImg,&minval,&maxval);
	cout<<"minval"<<minval<<endl;
	cout<<"maxval"<<maxval<<endl;
	cv::waitKey(0);
	return 0;
}
