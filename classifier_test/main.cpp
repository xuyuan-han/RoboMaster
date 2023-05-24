#include <opencv2/opencv.hpp>
#include <timer.hpp>
#include <classifier.hpp>

int main()
{
    classifierTrainer myTest("../image/Template_Image/*.png");
    sp::timer timer; //建立计时器
    timer.reset(); // 开始计时

    myTest.compare("../image/Test_Image/*.png");
    // myTest.compare("../image/Temperate/*.png");
    
    std::cout<<std::endl;
    std::cout<<"> myTest.compare运行时间："<<timer.get()<<"ms"<< std::endl; //结束计时
    std::cout<<"total:"<<myTest.total<<std::endl;
    std::cout<<"good:"<<myTest.good<<std::endl;
    std::cout<<"bad:"<<myTest.bad<<std::endl;
    std::cout<<"positive："<<std::setprecision(3)<<static_cast<double>(myTest.good*1.0/myTest.total)*100<<"%"<<std::endl;
    std::cout<<"negative："<<std::setprecision(3)<<static_cast<double>(myTest.bad*1.0/myTest.total)*100<<"%"<<std::endl;
    std::cout<<"test for the negative"<<std::endl;

    return 0;
}
