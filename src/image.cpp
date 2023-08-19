#include "image.h"

Image::Image(std::string _path, int _imageID) : path(_path), imageID(_imageID) {
	this -> loadImageData();
	memset(this -> signature, '\0', 3);
	memcpy(this -> signature, this -> fileHeader.signature, 2);
}

void Image::getHeaderInfo() {
	std::cout << "File signature: " << this -> signature 				<< std::endl;
	std::cout << "File size: " 		<< this -> fileHeader.fileSize		<< std::endl;
	std::cout << "File reserve: " 	<< this -> fileHeader.reserved 		<< std::endl;
	std::cout << "Data offset: " 	<< this -> fileHeader.dataOffset 	<< std::endl;
}

void Image::getImageInfo() {
	std::cout << "Image width: " 		<< this -> infoHeader.width 		 << std::endl;
	std::cout << "Image height: " 		<< this -> infoHeader.height		 << std::endl;
	std::cout << "Bytes per pixel: " 	<< this -> infoHeader.bitsPerPixel 	 << std::endl;
	std::cout << "Image size: " 		<< this -> infoHeader.imageSize 	 << std::endl;
}

void Image::hexdump(){
	for(int i = 0; i < this -> infoHeader.width; i++){
		for(int j = 0; j < this -> infoHeader.height; j++){
			std::cout << std::hex << pixels[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void Image::testing(){
	const std::string table = " .:-=+*#%@";
	for(int i = 0; i < this -> infoHeader.height; i++){
		for(int j = 0; j < this -> infoHeader.width; j++){
			int32_t idx = convertGrayScale(pixels[i][j]);
			idx = std::round((Scalar(idx) / 255) * 9);
			std::cout << table[idx] << table[idx];
		}
		std::cout << std::endl;
	}
}

int Image::convertGrayScale(int pixel){
	int blue = pixel & 0xff;
	int green = (pixel >> 8) & 0xff;
	int red = (pixel >> 16) & 0xff;
	return std::round(Scalar(blue + green + red) / 3);
}

Scalar Image::getFloatValue(int pixel){
	if(this -> infoHeader.bitsPerPixel == 24) {
		int blue = pixel & 0xff;
		int green = (pixel >> 8) & 0xff;
		int red = (pixel >> 16) & 0xff;
		return Scalar(blue + green + red) / 3;
	} else if (this -> infoHeader.bitsPerPixel == 8){
		int blue = (pixel & 0b11100000) >> 5;
		int green = (pixel & 0b00011100) >> 2;
		int red = (pixel &0b00000011);
		return Scalar(blue + green + red) / 3;
	} else return 0.0f;
}

std::vector<Scalar> Image::getPixelArray() {
	std::vector<Scalar> ret(this -> infoHeader.width * this -> infoHeader.height);
	for(int i = 0; i < this -> infoHeader.height; i++){
		for(int j = 0; j < this -> infoHeader.width; j++){
			Scalar factor = 1.0f;
			if(this -> infoHeader.bitsPerPixel == 24) factor = 255.0f;
			else if (this -> infoHeader.bitsPerPixel == 8) factor = 8.0f;
			if(this -> invert)
				ret[i * this -> infoHeader.width + j] = 1.0f - getFloatValue(pixels[i][j]) / factor;
			else 
				ret[i * this -> infoHeader.width + j] = getFloatValue(pixels[i][j]) / factor;
		}
	}
	return ret;
}

void Image::loadImageData() {
	(this -> file).open(this -> path, std::ios::in | std::ios::binary);
	try{
		if(!file.is_open()) throw Image::FileNotFound();
		else 
		{
			/* Reading file header section */
			this -> file.seekg(0, std::ios::beg);
			this -> file.read(reinterpret_cast<char*>(&this -> fileHeader), 14);
				
			/* Reading file infoheader section */
			this -> file.read(reinterpret_cast<char*>(&this -> infoHeader), 40);
			
			//image type check
			if(this -> infoHeader.colorPlanes != 1) 	 throw Image::UnsupportedFormat();
			if(this -> infoHeader.bitsPerPixel / 8 == 3)
			{
				//loading image pixels dat
				this -> file.seekg(this -> fileHeader.dataOffset, std::ios::beg);
				int bytesPerPixel = this -> infoHeader.bitsPerPixel / 8;
				int padding = (4 - ((this -> infoHeader.width * bytesPerPixel) % 4)) % 4;

				//Loading the entire image pixels data into Buffer
				char * Buffer = new char[this -> infoHeader.imageSize];
				file.read(Buffer, this -> infoHeader.imageSize);
				std::vector<std::vector<int>> _pixels(this -> infoHeader.height, std::vector<int>(this -> infoHeader.width));

				int dataIndex = 0;
				for (int i = this -> infoHeader.height - 1; i >= 0; i--) {
					for (int j = 0; j < this -> infoHeader.width; j++) {
						int offset = j * bytesPerPixel;
						int blue = Buffer[dataIndex + offset] & 0xFF;
						int green = Buffer[dataIndex + offset + 1] & 0xFF;
						int red = Buffer[dataIndex + offset+ 2] & 0xFF;
						_pixels[i][j] = (red << 16) | (green << 8) | blue;
					}
					dataIndex +=  bytesPerPixel * this -> infoHeader.width + padding;
				}
				this -> pixels = _pixels;
				delete [] Buffer;
			} else if (this -> infoHeader.bitsPerPixel / 8 == 1) {
				// Loading the entire image pixels data into Buffer
				this -> file.seekg(this -> fileHeader.dataOffset, std::ios::beg);
				char * Buffer = new char[this -> infoHeader.height * this -> infoHeader.width];
				file.read(Buffer, this -> infoHeader.height * this -> infoHeader.width);
				std::vector<std::vector<int>> _pixels(this -> infoHeader.height, std::vector<int>(this -> infoHeader.width));
				uint32_t dataindex = 0;
				for(int i = this -> infoHeader.height - 1; i >= 0; i--){
					for(int j = 0; j < this -> infoHeader.width; j++){
						_pixels[i][j] = (int)Buffer[dataindex + j];
					}
					dataindex += this -> infoHeader.width;
				}
				this -> pixels = _pixels;
				delete [] Buffer;
			} else throw Image::UnsupportedFormat();
			// Close the file when finished
			file.close();
		}
	} catch(std::exception &e) {
		std::cout << e.what() << std::endl;
		exit(-1);
	}
}

std::vector<Scalar> Image::extractPixelData(std::string path){
	Image img(path);
	return img.getPixelArray();
}