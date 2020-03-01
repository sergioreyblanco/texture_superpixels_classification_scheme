
/**
			  * @file				pca.h
			  *
				* @author			-
				*
			  * @brief      PCA variables extraction
			  */

#include <vector>
#include <eigen3/Eigen/Dense>


template<typename Type>
class pca_t
{
public:

	enum class sort_order : uint8_t
	{
		none, ascending, descending
	};

	pca_t() = default;
	~pca_t() = default;

	void set_input(const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& _input_matrix)
	{
		input_matrix = _input_matrix;
	}


	void compute(sort_order order = sort_order::ascending)
	{
		// Compute a centered version of input matrix
		centered_matrix = input_matrix.rowwise() - input_matrix.colwise().mean();
		mean_matrix = input_matrix - centered_matrix;
		//
		// Compute covariance matrix
		covariance_matrix = (centered_matrix.adjoint() * centered_matrix) / (Type)(input_matrix.rows() - 1);
		//
		// Use SelfAdjointEigenSolver to get eigen values and eigen vectors
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>> eigen_solver(covariance_matrix);
		eigen_values = eigen_solver.eigenvalues();
		eigen_vectors = eigen_solver.eigenvectors();

		if (order != sort_order::none)
			sort_eigen_vectors(order);

		// Projection is W * X'
		const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& W = eigen_vectors.adjoint();
		projection_matrix = W.topRows(input_matrix.cols()) * centered_matrix.adjoint();
	}


	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> reprojection()
	{
		return projection_matrix.transpose() * eigen_vectors + get_mean();
		//return (eigen_vectors * projection_matrix + get_mean().transpose()).transpose();
	}


	const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& get_input_matrix() const { return input_matrix; }
	const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& get_centered_matrix() const { return centered_matrix; }
	const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& get_covariance_matrix() const { return covariance_matrix; }
	const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& get_projection_matrix() const { return projection_matrix; }
	const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& get_mean() const { return mean_matrix; }

	const Eigen::Matrix<Type, 1, Eigen::Dynamic>& get_eigen_values() const { return eigen_values; };
	const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& get_eigen_vectors() const { return eigen_vectors; }


private:

	//
	// Private Methods
	//

	void sort_eigen_vectors(sort_order order = sort_order::ascending)
	{
		// Stuff below is done to sort eigen values. This can be done in other ways too.
		std::vector<std::pair<int, int>> eigen_value_index_vector;
		for (int i = 0; i < eigen_values.size(); ++i)
		{
			eigen_value_index_vector.push_back(std::make_pair(eigen_values[i], i));
		}

		if (order == sort_order::ascending)
			std::sort(std::begin(eigen_value_index_vector), std::end(eigen_value_index_vector), std::greater<std::pair<int, int>>());
		else
			std::sort(std::begin(eigen_value_index_vector), std::end(eigen_value_index_vector), std::less<std::pair<int, int>>());

		auto sorted_eigen_values = Eigen::Matrix<Type, 1, Eigen::Dynamic>(eigen_values.cols());
		auto sorted_eigen_vectors = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>(eigen_vectors.rows(), eigen_vectors.cols());
		for (int i = 0; i < eigen_values.size(); ++i)
		{
			sorted_eigen_values[i] = eigen_values[eigen_value_index_vector[i].second]; //can also be eigen_value_index_vector[i].first
			sorted_eigen_vectors.col(i) = eigen_vectors.col(eigen_value_index_vector[i].second);
		}

		eigen_values = sorted_eigen_values;
		eigen_vectors = sorted_eigen_vectors;
	}


	//
	// Private Attributes
	//

	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> input_matrix;
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> centered_matrix;
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> mean_matrix;
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix;
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> projection_matrix;

	Eigen::Matrix<Type, 1, Eigen::Dynamic> eigen_values;
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> eigen_vectors;

};
