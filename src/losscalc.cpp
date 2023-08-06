#include <losscalc.h>

// mean square error loss function
Scalar MeanSquareError(Matrix * vec, Matrix * expected){
	RowVector pred = *(RowVector*)vec;
	RowVector expc = *(RowVector*)expected;
	return 0.5 * (pred - expc).dot(pred - expc);
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
// Scalar 	BinaryCrossEntropy(Matrix * vec, Matrix * expected) {
// 	return (Matrix((( - *expected).array() * (vec -> unaryExpr([](Scalar x) -> Scalar { return log(x);})).array())) - Matrix((expected -> unaryExpr([](Scalar x) -> Scalar { return 1 - x;})).array() * (vec -> unaryExpr([](Scalar x) -> Scalar { return log(1 - x);})).array())).mean();
// }

Scalar 	BinaryCrossEntropy(Matrix * vec, Matrix * expected) {
	return Matrix((( - *expected).array() * (vec -> unaryExpr([](Scalar x) -> Scalar { return log(x);})).array())).mean();
}

std::vector<Matrix *> dBinaryCrossEntropy(std::vector<Matrix *> * output, std::vector<Matrix *> * expected)
{
	std::vector<Matrix *> errors;
	for(size_t i = 0; i < output -> size(); i++)
	{
		errors.push_back(new Matrix((*expected)[i] -> rows(), (*expected)[i] -> cols()));
		Matrix y_true_t = (*expected)[i] -> unaryExpr([](Scalar x) -> Scalar {return 1 - x;});
		Matrix y_pred_t = (*output)[i] -> unaryExpr([](Scalar x) -> Scalar {return 1 - x;});
		*errors[i] = - (Matrix(y_true_t.array() / y_pred_t.array() - (*expected)[i] -> array() / (*output)[i] -> array()))/((*output)[i] -> cols() * (*output)[i] -> rows());
	}
	return errors;
}