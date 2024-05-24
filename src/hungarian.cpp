#include "hungarian.h"
#include <limits>
#include <algorithm>

using namespace std;
using namespace Eigen;

// HungarianAlgorithm::HungarianAlgorithm(const std::vector<std::vector<double>> &costMatrix)
//     : costMatrix(costMatrix), n(costMatrix.size()), m(costMatrix[0].size())
// {
//     u.assign(n, 0);
//     v.assign(m, 0);
//     p.assign(m, -1);
//     way.assign(m, -1);
// }

// double HungarianAlgorithm::solve(std::vector<int> &result)
// {
//     result.assign(n, -1);
//     for (int i = 0; i < n; ++i)
//     {
//         std::vector<double> minv(m, INF);
//         std::vector<bool> used(m, false);
//         int j0 = -1, j1 = 0;
//         p[0] = i;
//         do
//         {
//             used[j1] = true;
//             int i0 = p[j1];
//             double delta = INF;
//             for (int j = 0; j < m; ++j)
//             {
//                 if (!used[j])
//                 {
//                     double cur = costMatrix[i0][j] - u[i0] - v[j];
//                     if (cur < minv[j])
//                     {
//                         minv[j] = cur;
//                         way[j] = j1;
//                     }
//                     if (minv[j] < delta)
//                     {
//                         delta = minv[j];
//                         j0 = j;
//                     }
//                 }
//             }
//             for (int j = 0; j <= m; ++j)
//             {
//                 if (used[j])
//                 {
//                     u[p[j]] += delta;
//                     v[j] -= delta;
//                 }
//                 else
//                 {
//                     minv[j] -= delta;
//                 }
//             }
//             j1 = j0;
//         } while (p[j1] != -1);
//         do
//         {
//             int j2 = way[j1];
//             p[j1] = p[j2];
//             j1 = j2;
//         } while (j1);
//     }
//     result.resize(n);
//     for (int j = 0; j < m; ++j)
//     {
//         if (p[j] != -1)
//         {
//             result[p[j]] = j;
//         }
//     }
//     return -v[0];
// }

double HungarianAlgorithm::Solve(const MatrixXd &costMatrix, vector<int> &assignment)
{
    int n = costMatrix.rows();
    int m = costMatrix.cols();

    if (n != m)
    {
        throw runtime_error("Cost matrix is not square!");
    }

    // Step 1: Subtract row minima
    MatrixXd cost = costMatrix;
    for (int i = 0; i < n; ++i)
    {
        double rowMin = cost.row(i).minCoeff();
        cost.row(i) -= VectorXd::Constant(n, rowMin);
    }

    // Step 2: Subtract column minima
    for (int j = 0; j < n; ++j)
    {
        double colMin = cost.col(j).minCoeff();
        cost.col(j) -= VectorXd::Constant(n, colMin);
    }

    // Cover all zeros with minimum number of lines
    vector<bool> coveredRows(n, false);
    vector<bool> coveredCols(n, false);
    vector<vector<bool>> starred(n, vector<bool>(n, false));
    vector<vector<bool>> primed(n, vector<bool>(n, false));

    // Step 3: Star zeros
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (cost(i, j) == 0 && !coveredRows[i] && !coveredCols[j])
            {
                starred[i][j] = true;
                coveredRows[i] = true;
                coveredCols[j] = true;
            }
        }
    }

    // Reset covered rows and columns
    fill(coveredRows.begin(), coveredRows.end(), false);
    fill(coveredCols.begin(), coveredCols.end(), false);

    // Step 4: Cover each column containing a starred zero
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (starred[i][j])
            {
                coveredCols[j] = true;
            }
        }
    }

    while (count(coveredCols.begin(), coveredCols.end(), true) < n)
    {
        // Step 5: Find a non-covered zero and prime it
        bool found = false;
        int z0Row = -1, z0Col = -1;
        for (int i = 0; i < n; ++i)
        {
            if (!coveredRows[i])
            {
                for (int j = 0; j < n; ++j)
                {
                    if (!coveredCols[j] && cost(i, j) == 0)
                    {
                        primed[i][j] = true;
                        // Step 5.1: If there is no starred zero in the row, go to Step 6
                        bool starInRow = false;
                        for (int k = 0; k < n; ++k)
                        {
                            if (starred[i][k])
                            {
                                starInRow = true;
                                coveredRows[i] = true;
                                coveredCols[k] = false;
                                break;
                            }
                        }
                        if (!starInRow)
                        {
                            z0Row = i;
                            z0Col = j;
                            found = true;
                            break;
                        }
                    }
                }
            }
            if (found)
                break;
        }

        // Step 6: Construct a series of alternating primed and starred zeros
        if (z0Row != -1 && z0Col != -1)
        {
            vector<pair<int, int>> path;
            path.emplace_back(z0Row, z0Col);

            while (true)
            {
                // Find starred zero in the column
                int row = -1;
                for (int i = 0; i < n; ++i)
                {
                    if (starred[i][path.back().second])
                    {
                        row = i;
                        break;
                    }
                }
                if (row == -1)
                    break;
                path.emplace_back(row, path.back().second);

                // Find primed zero in the row
                int col = -1;
                for (int j = 0; j < n; ++j)
                {
                    if (primed[path.back().first][j])
                    {
                        col = j;
                        break;
                    }
                }
                path.emplace_back(path.back().first, col);
            }

            // Unstar each starred zero of the series, and star each primed zero
            for (const auto &p : path)
            {
                if (starred[p.first][p.second])
                {
                    starred[p.first][p.second] = false;
                }
                else
                {
                    starred[p.first][p.second] = true;
                }
            }

            // Reset covered rows and columns and erase all prime marks
            fill(coveredRows.begin(), coveredRows.end(), false);
            fill(coveredCols.begin(), coveredCols.end(), false);
            for (auto &row : primed)
            {
                fill(row.begin(), row.end(), false);
            }

            // Cover each column containing a starred zero
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (starred[i][j])
                    {
                        coveredCols[j] = true;
                    }
                }
            }
        }
        else
        {
            // Step 7: Adjust the cost matrix
            double minUncovered = numeric_limits<double>::max();
            for (int i = 0; i < n; ++i)
            {
                if (!coveredRows[i])
                {
                    for (int j = 0; j < n; ++j)
                    {
                        if (!coveredCols[j])
                        {
                            minUncovered = min(minUncovered, cost(i, j));
                        }
                    }
                }
            }

            for (int i = 0; i < n; ++i)
            {
                if (coveredRows[i])
                {
                    for (int j = 0; j < n; ++j)
                    {
                        cost(i, j) += minUncovered;
                    }
                }
            }

            for (int j = 0; j < n; ++j)
            {
                if (!coveredCols[j])
                {
                    for (int i = 0; i < n; ++i)
                    {
                        cost(i, j) -= minUncovered;
                    }
                }
            }
        }
    }

    // Extract the assignment
    assignment.resize(n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (starred[i][j])
            {
                assignment[i] = j;
                break;
            }
        }
    }

    // Calculate the cost
    double costSum = 0;
    for (int i = 0; i < n; ++i)
    {
        costSum += costMatrix(i, assignment[i]);
    }

    return costSum;
}
