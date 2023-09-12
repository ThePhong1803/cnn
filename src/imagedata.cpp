#include <imagedata.h>

/* image wrapper class implementation */
// default constructor
ImageData::ImageData() 
{
    this -> number = -1;
    this -> id     = -1;
}                                    

// constructor with pixel vector
ImageData::ImageData(std::vector<Scalar> _pixels, int _number, int _id)
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
	// img.getImageInfo();
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
            if(this -> pixels[r * mat -> cols() + c] > 1.0f || this -> pixels[r * mat -> cols() + c] < 0.0f) {
                std::cout << "Error Element Found: " << this -> pixels[r * mat -> cols() + c] << std::endl;
            }
            mat -> coeffRef(r, c) = this -> pixels[r * mat -> cols() + c];
        }
    }
    // normalize data (dividing every element by the in matrix ole length)
    mat -> normalize();
    // Scalar const std_dev = sqrt((mat -> array() - mat -> mean()).square().sum() / (mat -> size() - 1));
    // Scalar const mean = mat -> mean();
    // *mat = mat -> unaryExpr([&mean, &std_dev](Scalar x) -> Scalar { return (x - mean) / std_dev;});
}