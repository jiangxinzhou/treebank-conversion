/*
 * Param.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef PARAM_H_
#define PARAM_H_

#include "Eigen/Dense"
#include "BaseParam.h"

// Notice: aux is an auxiliary variable to help parameter updating
class Param : public BaseParam {
  public:
    Tensor2D aux_square;
    Tensor2D aux_mean;
    int iter;

	inline void initial(int outDim, int inDim) {
		val.init(outDim, inDim);
		grad.init(outDim, inDim);
		aux_square.init(outDim, inDim);
		aux_mean.init(outDim, inDim);
		//val.mat() = val.mat().unaryExpr(ptr_fun(normal_distribution_0));
		//val.mat() = val.mat() / sqrt(inDim);
		iter = 0;
		MatrixXd I = MatrixXd::Identity(inDim, inDim);
		double lr = .1;
		double eps = .05 / (outDim + inDim);
		bool success = false;
		int tries = 0;
		while (!success && tries < 10) {
			val.mat() = val.mat().unaryExpr(ptr_fun(normal_distribution_0));
			val.mat() = val.mat() / sqrt(inDim);
			for (int i = 0; i < 100; i++) {
				MatrixXd QTQmI = val.mat().transpose()* val.mat() - I;
				MatrixXd Q2 = val.mat().array().square();
				MatrixXd losstemp = QTQmI.array().square() / 2;
				MatrixXd Qx = MatrixXd::Zero(outDim, 1);
				MatrixXd Qy = MatrixXd::Zero(1, inDim);
				double loss = 0.0;
				for (int i = 0; i < losstemp.rows(); i++) {
					for (int j = 0; j < losstemp.cols(); j++) {
						loss += losstemp(i, j);
					}
				}
				for (int i = 0; i < Q2.rows(); i++) {
					double sum = 0.0;
					for (int j = 0; j < Q2.cols(); j++) {
						sum += Q2(i, j);
					}
					Qx(i, 0) = sum;
				}
				for (int i = 0; i < Q2.cols(); i++) {
					double sum = 0.0;
					for (int j = 0; j < Q2.rows(); j++) {
						sum += Q2(j, i);
					}
					Qy(0, i) = sum;
				}
				for (int i = 0; i < Q2.rows(); i++) {
					for (int j = 0; j < Q2.cols(); j++) {
						Q2(i, j) = abs(Q2(i, j) + Qx(i, 0) + Qy(0, j) - 1) + eps;
					}
				}
				MatrixXd temp = lr * val.mat() * QTQmI;
				for (int i = 0; i < temp.rows(); i++) {
					for (int j = 0; j < temp.cols(); j++) {
						val.mat()(i, j) -= temp(i, j) / Q2(i, j);
					}
				}
				double max = -1e20;
				for (int i = 0; i < Q2.rows(); i++) {
					for (int j = 0; j < Q2.cols(); j++) {
						if (max < val.mat()(i, j))
							max = val.mat()(i, j);
					}
				}
				if (max > 1e6 || loss > 1e6 || loss != INFINITY) {
					tries += 1;
					lr /= 2;
					break;
				}
			}
			success = true;
		}
		if (!success) {
			cout << "orthonormal initializer error" << endl;
			val.mat() = val.mat().unaryExpr(ptr_fun(normal_distribution_0));
			val.mat() = val.mat() / sqrt(inDim);
		}
	}
	
	
	inline void initial_constant(int outDim, int inDim) {
		val.init(outDim, inDim);
		grad.init(outDim, inDim);
		aux_square.init(outDim, inDim);
		aux_mean.init(outDim, inDim);
		val = 0;
		iter = 0;
	}
    // allow sparse and dense parameters have different parameter initialization methods
    // inline void initial(int outDim, int inDim) {
        // val.init(outDim, inDim);
        // grad.init(outDim, inDim);
        // aux_square.init(outDim, inDim);
        // aux_mean.init(outDim, inDim);

        // dtype bound = sqrt(6.0 / (outDim + inDim + 1));
        // val.random(bound);
        // iter = 0;
    // }

    inline int outDim() {
        return val.row;
    }

    inline int inDim() {
        return val.col;
    }

    inline void clearGrad() {
        grad.zero();
    }

    inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_square.vec() = aux_square.vec() + grad.vec().square();
        val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
    }

    inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
		if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
		grad.vec() = aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
        /*val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();*/
        iter++;
    }

    inline void randpoint(int& idx, int &idy) {
        //select indexes randomly
        std::vector<int> idRows, idCols;
        idRows.clear();
        idCols.clear();
        for (int i = 0; i < val.row; i++)
            idRows.push_back(i);
        for (int i = 0; i < val.col; i++)
            idCols.push_back(i);

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idy = idRows[0];
        idx = idCols[0];
    }

    inline dtype squareGradNorm() {
        dtype sumNorm = 0.0;
        for (int i = 0; i < grad.size; i++) {
            sumNorm += grad.v[i] * grad.v[i];
        }
        return sumNorm;
    }

    inline void rescaleGrad(dtype scale) {
        grad.vec() = grad.vec() * scale;
    }

    inline void save(std::ofstream &os)const {
        val.save(os);
        aux_square.save(os);
        aux_mean.save(os);
        os << iter << endl;
    }

    inline void load(std::ifstream &is) {
        val.load(is);
        aux_square.load(is);
        aux_mean.load(is);
        is >> iter;
    }
};

#endif /* PARAM_H_ */
