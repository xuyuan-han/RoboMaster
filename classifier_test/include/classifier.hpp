// #define FRAME_BY_FRAME
// #define SHOW_TEMPLATE_IMAGE_FRAME_BY_FRAME

// #define DEBUG_CLASSIFIER
// #define DEBUG_PRINT_ARMORNUM
// #define DEBUG_CLASSIFIER_ORB

// #define SHOW_CLASSIFIER_IMAGE
// #define DRAW_IMAGE_FEATURE_MATCH

// #define PRINT_CLASSIFIER_RUNTIME

#define CLASSIFIER_OUTPUT

#define CLASSIFIER_IMAGEPART_ROWS 100
#define CLASSIFIER_IMAGEPART_COLS 120
#define THRESH_BINAR_TEMPLATE 0.31 //二值化template_image取thresh_binar最亮部分
#define THRESH_BINAR_TEST 0.35 //二值化test_image取thresh_binar最亮部分


#include <opencv2/xfeatures2d.hpp>
#include <timer.hpp>
#include <get_proportion_thresh.hpp>

class classifierTrainer
{
private:
    /* data */
public:
    std::string template_img_location;
    std::vector<cv::String> template_image_names;
    std::vector<cv::Mat> template_images;
    std::string test_img_location;
    std::vector<cv::String> test_image_names;
    std::string count_classifier_str; //用于输出图像的命名

    int total, good=0, bad=0;
    int maxGainArmor = -1;

    classifierTrainer(std::string template_img_location);
    ~classifierTrainer();
    void compare(std::string test_img_loc);
    bool ORB_classifier_isok(const cv::Mat& img2,int img_num);
};

classifierTrainer::classifierTrainer(std::string template_img_loc):template_img_location(template_img_loc)
{
    #ifdef DEBUG_CLASSIFIER
    std::cout<<"ClassifierTrainer is being created"<<std::endl;
    #endif

    cv::glob(template_img_location, template_image_names); //将文件名放入template_image_names容器
    for(int i=0;i<template_image_names.size();i++) //循环遍历所有文件
    {
        cv::Mat template_image = imread(template_image_names[i],0);
        
        #ifdef DEBUG_CLASSIFIER
        std::cout << "读入" << template_image_names[i] << "号装甲板模板" << std::endl;
        #endif

        if(!template_image.data)//检测图片是否读取成功
        {        
            std::cout<<"读取第"<<i+1<<"张图片错误，请确定目录下是否存在该图片"<<std::endl;
        }
        
        int threshold_int = sp::get_proportion_thresh(template_image, THRESH_BINAR_TEMPLATE); //二值化模板图像

        #ifdef DEBUG_CLASSIFIER
        std::cout<<"template threshold_int="<<threshold_int<<std::endl;
        #endif
        
        if(threshold_int!=0)
        {
            cv::threshold(template_image, template_image, threshold_int, 255, CV_THRESH_BINARY); //二值化模板图像
        }

        cv::medianBlur(template_image, template_image, 3); //中值滤波

		cv::resize(template_image, template_image, cv::Size(CLASSIFIER_IMAGEPART_COLS, CLASSIFIER_IMAGEPART_ROWS), (0,0), (0,0), CV_INTER_AREA); // 将模板图像的大小变成CLASSIFIER_IMAGEPART_COLS*CLASSIFIER_IMAGEPART_ROWS

   		template_images.push_back(template_image);
    }

    #ifdef SHOW_TEMPLATE_IMAGE_FRAME_BY_FRAME
    for(int i=0;i<template_images.size();i++)
    {
        cv::imshow("img",template_images[i]);
        cv::waitKey(0);
    }
    #endif
}

classifierTrainer::~classifierTrainer()
{
    #ifdef DEBUG_CLASSIFIER
    std::cout<<"ClassifierTrainer is being deleted"<<std::endl;
    #endif
}

void classifierTrainer::compare(std::string test_img_loc)
{
//--------------------------------------------------将文件名放入test_image_names容器----------------------------------------
    test_img_location = test_img_loc;
    cv::glob(test_img_location, test_image_names);

//--------------------------------------------------变量声明---------------------------------------------------------------

    std::vector<int> gain_list; //声明容器gain_list来放置每个图像的gain

    total = test_image_names.size();

    for(int i=0;i<test_image_names.size();i++) //循环遍历所有文件,开始分类
    {
//----------------------------------------------------预处理测试图像--------------------------------------------------
        #ifdef DEBUG_CLASSIFIER
        std::cout<<std::endl;
        std::cout<<"------------------------------------------"<<std::endl;
        std::cout<<">> "<<test_image_names[i]<<std::endl;
        std::cout<<"maxGainArmor初始化:"<<maxGainArmor<<std::endl;
        #endif

        cv::Mat test_image = imread(test_image_names[i],0);
        cv::Mat test_image_copy;
        test_image.copyTo(test_image_copy);

        int threshold_int;
        threshold_int = sp::get_proportion_thresh(test_image, THRESH_BINAR_TEST); //二值化测试图像
        
        #ifdef DEBUG_CLASSIFIER
        std::cout<<"test threshold_int="<<threshold_int<<std::endl;
        #endif

        if(threshold_int!=0)
        {
            cv::threshold(test_image, test_image, threshold_int, 255, CV_THRESH_BINARY);
        }

        cv::medianBlur(test_image, test_image, 11); //中值滤波
        
        cv::resize(test_image, test_image, cv::Size(CLASSIFIER_IMAGEPART_COLS, CLASSIFIER_IMAGEPART_ROWS), (0,0), (0,0), CV_INTER_AREA); // 将测试图像的大小变成CLASSIFIER_IMAGEPART_COLS*CLASSIFIER_IMAGEPART_ROWS

        #ifdef SHOW_CLASSIFIER_IMAGE
        cv::imshow("CLASSIFIER_IMAGE", test_image);
        #endif
//----------------------------------------------------预处理测试图像完成-----------------------------------------------

        #ifdef PRINT_CLASSIFIER_RUNTIME
        sp::timer timer_classifier; //建立计时器
        timer_classifier.reset(); // 开始计时
        #endif

//--------------------------------------------------------开始分类------------------------------------------------------------
        #ifdef DEBUG_CLASSIFIER
        std::cout << "开始分类" << std::endl;
        #endif

        int gain = 0; //初始化gain
    
//----------------------------------------------- 逐像素获取每个像素的gain并累积------------------------------------------------------------
        for(int template_count=0; template_count<template_images.size(); template_count++)
        {
            #ifdef DEBUG_CLASSIFIER
            std::cout << "遍历第" << template_count+1 << "个模板" << std::endl;
            #endif

            for(int i=0; i<CLASSIFIER_IMAGEPART_ROWS; i++)
            {
                //获取第i行首像素指针
                uchar* p_template_image = template_images[template_count].ptr<uchar>(i);
                uchar* p_test_image = test_image.ptr<uchar>(i);

                //对第i行的每个像素（Byte）进行操作
                for(int j=0; j<CLASSIFIER_IMAGEPART_COLS; j++)
                {
                    //用指针访问像素的方法（速度快）
                    if(p_template_image[j]==255 && p_test_image[j]==255)
                    {
                        gain += 3;
                    }
                    else if(p_template_image[j] != p_test_image[j])
                    {
                        gain -= 2;
                    }
                    else{}
                }
            }
        gain_list.push_back(gain); //将gain加入gain_list

        #ifdef DEBUG_CLASSIFIER
        std::cout << template_count+1 << "号装甲板的gain是" << gain << std::endl; //打印gain
        #endif

        gain = 0; //重置gain
        }

        auto min = std::min_element(gain_list.begin(), gain_list.end());
        auto max = std::max_element(gain_list.begin(), gain_list.end());

        #ifdef DEBUG_CLASSIFIER
        std::cout << "这组图像的最小gain是" << *min << std::endl;
        std::cout << "这组图像的最大gain是" << *max << std::endl;
        #endif

//--------------------------------------获取当前时间作为imwrite文件名----------------------------------------
        std::string filePath;
        filePath.clear();
        sp::timer timer_now;
        long long int count_classifier_int(timer_now.getTimeStamp());
        count_classifier_str.clear();
        count_classifier_str = std::to_string(count_classifier_int);

        if(*max<500)
        {
            #ifdef DEBUG_CLASSIFIER
            std::cout << "舍弃" << std::endl;
            #endif

            #ifdef CLASSIFIER_OUTPUT
            filePath = "../image/dst/negative/negative_1_"+test_image_names[i].substr(14)+"_"+count_classifier_str+".jpg";
            cv::imwrite(filePath, test_image_copy);

            #ifdef DEBUG_CLASSIFIER
            std::cout << ">> 输出negative图片成功" << std::endl;
            #endif

            #endif

            #ifdef PRINT_CLASSIFIER_RUNTIME
            std::cout << std::endl;
            std::cout << "> 一级分类器运行时间：" << timer_classifier.get() << "ms" << std::endl; //结束计时
            #endif

            #ifdef FRAME_BY_FRAME
            cv::waitKey(0);
            #endif

            bad++;
            maxGainArmor = -1;
            gain_list.clear();
        }
        else
        {
            maxGainArmor = (max_element(gain_list.begin(),gain_list.end()) - gain_list.begin()) + 1;

            #ifdef DEBUG_PRINT_ARMORNUM
            if(maxGainArmor >= 0)
            {
                std::cout << ">> 一级分类器检测结果对应编号为" << maxGainArmor << "的装甲板" << std::endl <<std::endl;
            }
            #endif

            #ifdef PRINT_CLASSIFIER_RUNTIME
            std::cout << std::endl;
            std::cout << "> 一级分类器运行时间：" << timer_classifier.get() << "ms" << std::endl; //结束计时
            #endif

//-----------------------------------------------引入二级ORB分类器------------------------------
            if(ORB_classifier_isok(test_image,i) //使用ORB分类器
            )
            {
                #ifdef DEBUG_CLASSIFIER_ORB
                std::cout << "> 一级分类器接受到ORB返回的true" << std::endl; 
                #endif

                #ifdef CLASSIFIER_OUTPUT
                filePath = "../image/dst/positive/positive_#"+std::to_string(maxGainArmor)+"_"+test_image_names[i].substr(14)+"_"+count_classifier_str+".jpg";
                cv::imwrite(filePath, test_image_copy);
                #ifdef DEBUG_CLASSIFIER
                std::cout << ">> 输出positive图片成功" << std::endl;
                #endif
                #endif

                #ifdef FRAME_BY_FRAME
                cv::waitKey(0);
                #endif

                good++;
                maxGainArmor = -1;
                gain_list.clear();
            }
            else
            {
                #ifdef CLASSIFIER_OUTPUT
                filePath = "../image/dst/negative/negative_2_"+test_image_names[i].substr(14)+"_"+count_classifier_str+".jpg";
                cv::imwrite(filePath, test_image_copy);
                #ifdef DEBUG_CLASSIFIER
                std::cout << ">> 输出negative图片成功" << std::endl;
                #endif
                #endif
                
                #ifdef DEBUG_CLASSIFIER_ORB
                std::cout << "> 一级分类器接受到ORB返回的false" << std::endl; 
                #endif
                
                #ifdef FRAME_BY_FRAME
                cv::waitKey(0);
                #endif
                
                bad++;
                maxGainArmor = -1;
                gain_list.clear();
            }
    }
    }  
}

bool classifierTrainer::ORB_classifier_isok(const cv::Mat& img2,int img_num)
{
    #ifdef PRINT_CLASSIFIER_RUNTIME
    sp::timer timer_classifier_orb; //建立计时器
    timer_classifier_orb.reset(); // 开始计时
	#endif

    try
    { 
    //【1】载入原图片并显示+载入的图像是二值化之后的图像
    //img1是模板图像，img2是待检测图
    cv::Mat img1 = this->template_images[this->maxGainArmor-1];
    if(!img1.data||!img2.data)//检测图片是否读取成功
    {
        std::cout<<"读取图片错误，请确定目录下是否存在该图片"<<std::endl;
    }

    //【2】定义需要用到的变量和类
    cv::Ptr<cv::ORB> detector = cv::ORB::create(200,1.2); //定义一个ORB特征检测类对象detector
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;//放置特征点
    cv::Mat descriptors_1, descriptors_2;

    //【3】调用detect函数检测出SURF特征关键点，保存在vector容器中
    detector->detectAndCompute(img1,cv::Mat(),keypoint_1,descriptors_1);
    detector->detectAndCompute(img2,cv::Mat(),keypoint_2,descriptors_2);
    
    #ifdef DEBUG_CLASSIFIER_ORB
    if(!descriptors_1.data)
    {
        std::cout<<"> descriptors_1无内容"<<std::endl;
    }
    if (!descriptors_2.data)
    {
        std::cout<<"> descriptors_2无内容"<<std::endl;
    }
    #endif
    
    //【4】基于FLANN的描述符对象匹配
    std::vector<cv::DMatch> matches;
    // 初始化flann匹配
    cv::flann::Index flannIndex(descriptors_1, cv::flann::LshIndexParams(12,20,2), cvflann::FLANN_DIST_HAMMING);

    //【10】匹配和测试描述符，获取两个最邻近的描述符
    cv::Mat matchIndex(descriptors_1.rows, 2, CV_32SC1);
    cv::Mat matchDistance(descriptors_1.rows, 2, CV_32FC1);

    flannIndex.knnSearch(descriptors_2, matchIndex, matchDistance, 2, cv::flann::SearchParams());//调用K邻近算法

    //【11】根据劳氏算法(Low's algorithm)选出优秀的匹配
    std::vector<cv::DMatch> good_matches;
    for(int i=0;i<matchDistance.rows;i++)
    {
        if(matchDistance.at<float>(i,0) < 0.6*matchDistance.at<float>(i,1))
        {
            cv::DMatch dmatches(i, matchIndex.at<int>(i,0), matchDistance.at<float>(i,0));
            good_matches.push_back(dmatches);
        }
    }

    //【12】绘制并显示匹配窗口
    cv::Mat img_matches;
    cv::drawMatches(img2,keypoint_2,img1,keypoint_2,good_matches,img_matches);


    // // 【13】输出相关匹配点信息
    // for(int i=0;i<good_matches.size();i++)
    // {
    //     std::cout<<"> 符合条件的匹配点 "<<i<<" 特征点1："<<good_matches[i].queryIdx<<" -- 特征点2："<<good_matches[i].trainIdx<<std::endl;
    // }

    #ifdef DEBUG_CLASSIFIER_ORB
    //【14】打印特征信息
    std::cout<<"> img1检测到特征点"<<keypoint_1.size()<<"个"<<std::endl;
    std::cout<<"> img2检测到特征点"<<keypoint_2.size()<<"个"<<std::endl;
    std::cout<<"> 共匹配到特征点"<<good_matches.size()<<"对"<<std::endl;
    #endif


    //【15】绘制特征图像
    #ifdef DRAW_IMAGE_FEATURE_MATCH
    cv::Mat img1_keypoint, img2_keypoint;
    cv::drawKeypoints(img1,keypoint_1,img1_keypoint);
    cv::drawKeypoints(img2,keypoint_2,img2_keypoint);
    cv::imshow("> 特征点检测效果图1",img1_keypoint);
    cv::imshow("> 特征点检测效果图2",img2_keypoint);
    cv::imshow("匹配效果图",img_matches);
    #endif

    if(good_matches.size()>0)
    {
        #ifdef PRINT_CLASSIFIER_RUNTIME
	    std::cout << "> 二级分类器运行时间：" << timer_classifier_orb.get() << "ms" << std::endl; //结束计时
		#endif

        #ifdef CLASSIFIER_OUTPUT
        std::string filePath_orb;
        filePath_orb.clear();
        filePath_orb = "../image/dst/orb/positiveMatch/positiveMatch_"+test_image_names[img_num].substr(14)+"_"+count_classifier_str+".jpg";
        cv::imwrite(filePath_orb, img_matches);
        #ifdef DEBUG_CLASSIFIER
        std::cout << ">> 输出positive orb图片成功" << std::endl;
        #endif
        #endif

        return true;
    }
    else
    {
        #ifdef PRINT_CLASSIFIER_RUNTIME
	    std::cout << "> 二级分类器运行时间：" << timer_classifier_orb.get() << "ms" << std::endl; //结束计时
		#endif

        #ifdef CLASSIFIER_OUTPUT
        std::string filePath_orb;
        filePath_orb.clear();
        filePath_orb = "../image/dst/orb/negativeMatch/negativeMatch_"+test_image_names[img_num].substr(14)+"_"+count_classifier_str+".jpg";
        cv::imwrite(filePath_orb, img_matches);
        #ifdef DEBUG_CLASSIFIER
        std::cout << ">> 输出negative orb图片成功" << std::endl;
        #endif
        #endif
        
        return false;
    }
    

    }
    catch (std::exception& e) 
    {
        #ifdef PRINT_CLASSIFIER_RUNTIME
	    std::cout << "> 二级分类器运行时间：" << timer_classifier_orb.get() << "ms" << std::endl; //结束计时
		#endif

        #ifdef DEBUG_CLASSIFIER_ORB
        std::cout << "> ORB分类器出错:" << std::endl; 
        std::cout << "> ORB返回false" << std::endl;
        #endif

        return false;
    } 
}
