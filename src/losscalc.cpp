#include <losscalc.h>

// mean square error loss function
Scalar MeanSquareError(RowVector * vec, RowVector * expected){
	return 0.5 * (*vec - *expected).dot(*vec - *expected);
}

std::vector<Matrix *> dMeanSquareError(std::vector<Matrix *> * output, std::vector<Matrix *> * expected){
	std::vector<Matrix *> errors;
	for(size_t i = 0; i < output -> size(); i++)
	{
		errors.push_back(new Matrix((*expected)[i] -> rows(), (*expected)[i] -> cols()));
		*errors[i] = *(*expected)[i] - *(*output)[i];
	}
	return errors;
}

// binary cross entropy loss function
Scalar 	BinaryCrossEntropy(RowVector * vec, RowVector * expected) {
	return - ((*expected) * (vec -> unaryExpr([](Scalar x) -> Scalar { return log(x);}))).sum();
}
std::vector<Matrix *> dBinaryCrossEntropy(std::vector<Matrix *> * output, std::vector<Matrix *> * expected)
{
	std::vector<Matrix *> errors;
	for(size_t i = 0; i < output -> size(); i++)
	{
		errors.push_back(new Matrix((*expected)[i] -> rows(), (*expected)[i] -> cols()));
		*errors[i] = *(*expected)[i] - *(*output)[i];
	}
	return errors;
}