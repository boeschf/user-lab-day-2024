#include "utility.hpp"

int main(int argc, char** argv) {
    try {
        check_mpi(MPI_Init(&argc, &argv), "MPI_Init failed");
        auto rank = [](){
            int rank;
            check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank failed");
            return rank;
        }();
        auto size = [](){
            int size;
            check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size failed");
            return size;
        }();

        check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier failed");
        check_mpi(MPI_Finalize(), "MPI_Finalize failed");
    }
    catch (const exception& e) {
        std::visit([](const auto& e){ std::cout << e.what() << std::endl; }, e);
        throw;
    }
}
