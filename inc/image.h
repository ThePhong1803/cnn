#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <string.h>
#include <stdint.h>
#pragma once


#pragma pack(push, 1)
struct BitmapFileHeader {
    char signature[2];
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
};

struct BitmapInfoHeader {
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t colorPlanes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t horizontalResolution;
    int32_t verticalResolution;
    uint32_t colorsUsed;
    uint32_t importantColors;
};
#pragma pack(pop)

class Image {
	// class image with basic atribbutes and method
	private:
	char 				signature[3];
	BitmapFileHeader    fileHeader;
	BitmapInfoHeader	infoHeader;
	std::ifstream 		file;						// File contain image data
	std::string 		label;						// Label for image for classification
	std::string 		path;						// Path to the image location
	int 				imageID;					// Image ID
	bool 				invert;						// invert image input graysacle
	std::vector<std::vector<int>>	pixels;			// Container for pixel array, include R, G, B value.
	public:
	Image() {}																	// Default constructor
	Image(std::string _path, int _imageID = 0);	  								// Constructor with image path and load image}
	~Image() {}
	void loadImageData();														// Load pixel data into the array;
	void getHeaderInfo();
	void getImageInfo();
	void hexdump();
	void testing();
	float getFloatValue(int pixel);
	void setInvert(bool _invert) { this -> invert = _invert; }
	std::vector<float> getPixelArray();
	int convertGrayScale(int pixel);
	int getWidth() { return this -> infoHeader.width; }
	int getHeight() { return this -> infoHeader.height; }

	private:
	// exception handing class 
	class UnsupportedFormat : public std::exception {
		friend class Image;
		std::string msg;
		public:
		UnsupportedFormat() {
			msg = "Unsupported Bitmap Image Format\n";
		}
		const char* what() const throw() {
			return msg.c_str();
		}
	};
	
	class FileNotFound : public std::exception {
		friend class Image;
		std::string msg;
		public:
		FileNotFound() {
			msg = "File not found\n";
		}
		const char* what() const throw() {
			return msg.c_str();
		}
	};
	
	public:
	// static method for only extract image infomation only
	static std::vector<float> extractPixelData(std::string path);
};
