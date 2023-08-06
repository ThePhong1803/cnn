#include <imagedata.h>

/* image wrapper class implementation */
// default constructor
ImageData::ImageData() 
{
    this -> number = -1;
    this -> id     = -1;
}                                    

// constructor with pixel vector
ImageData::ImageData(std::vector<float> _pixels, int _number, int _id)
{
    this -> pixels = _pixels;
    this -> number = _number;
    this -> id     = _id;
}

// constructor with image file path
ImageData::ImageData(std::string path, bool invert, int _number, int _id)
{
    Image img(path);
    img.setInvert(invert);
    this -> number = _number;
    this -> id     = _id;
    this -> height = img.getHeight();
    this -> width  = img.getWidth();
    this -> pixels = img.getPixelArray();
}

// destructor
ImageData::~ImageData() 
{
    // do nothing
}

// method to copy pixel array to vector
void ImageData::getPixelVector(RowVector * temp)
{
	for(uint32_t i = 0; i < pixels.size(); i++)
    {
		(*temp)[i] = pixels[i];
	}
}


// method to copy pixel array to matrix
void ImageData::getPixelMatrix(Matrix * mat)
{
    // copy each pixel data into matrix
    for(int32_t r = 0; r < mat -> rows(); r++)
    {
        for(int32_t c = 0; c < mat -> cols(); c++)
        {
            mat -> coeffRef(r, c) = this -> pixels[r * mat -> cols() + c];
        }
    }
    // normalize data
    mat -> normalize();
}