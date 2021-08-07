/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Natasha Sharma, University of Texas at El Paso,
 *          Guido Kanschat, University of Heidelberg
 *          Timo Heister, Clemson University
 *          Wolfgang Bangerth, Colorado State University
 *          Zhuroan Wang, Colorado State University
 */


// @sect3{Include files}

// The first few include files have already been used in the previous
// example, so we will not explain their meaning here again. The principal
// structure of the program is very similar to that of, for example, step-4
// and so we include many of the same header files.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// The two most interesting header files will be these two:
#include <deal.II/fe/fe_interface_values.h>

#include <deal.II/meshworker/mesh_loop.h>
// The first of these is responsible for providing the class FEInterfaceValues
// that can be used to evaluate quantities such as the jump or average
// of shape functions (or their gradients) across interfaces between cells.
// This class will be quite useful in evaluating the penalty terms that appear
// in the C0IP formulation.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <cmath>
#include <fstream>
#include <iostream>


namespace Step47
{
  using namespace dealii;


  // In the following namespace, let us define the exact solution against
  // which we will compare the numerically computed one. It has the form
  // $u(x,y) = \sin(\pi x) \sin(\pi y)$ (only the 2d case is implemented),
  // and the namespace also contains a class that corresponds to the right
  // hand side that produces this solution.
  namespace ExactSolution
  {
    using numbers::PI;

    template <int dim>
    class Solution : public Function<dim>
    {
    public:
      static_assert(dim == 2, "Only dim==2 is implemented.");

      virtual double value(const Point<dim> &p,
                           const unsigned int /*component*/ = 0) const override
      {
        return std::sin(PI * p[0]) * std::sin(PI * p[1]);
      }

      virtual Tensor<1, dim>
      gradient(const Point<dim> &p,
               const unsigned int /*component*/ = 0) const override
      {
        Tensor<1, dim> r;
        r[0] = PI * std::cos(PI * p[0]) * std::sin(PI * p[1]);
        r[1] = PI * std::cos(PI * p[1]) * std::sin(PI * p[0]);
        return r;
      }

      virtual void
      hessian_list(const std::vector<Point<dim>> &       points,
                   std::vector<SymmetricTensor<2, dim>> &hessians,
                   const unsigned int /*component*/ = 0) const override
      {
        for (unsigned i = 0; i < points.size(); ++i)
          {
            const double x = points[i][0];
            const double y = points[i][1];

            hessians[i][0][0] = -PI * PI * std::sin(PI * x) * std::sin(PI * y);
            hessians[i][0][1] = PI * PI * std::cos(PI * x) * std::cos(PI * y);
            hessians[i][1][1] = -PI * PI * std::sin(PI * x) * std::sin(PI * y);
          }
      }
    };


    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      static_assert(dim == 2, "Only dim==2 is implemented");

      virtual double value(const Point<dim> &p,
                           const unsigned int /*component*/ = 0) const override

      {
        return 4 * std::pow(PI, 4.0) * std::sin(PI * p[0]) *
               std::sin(PI * p[1]);
      }
    };
  } // namespace ExactSolution



  // @sect3{The main class}
  //
  // The following is the principal class of this tutorial program. It has
  // the structure of many of the other tutorial programs and there should
  // really be nothing particularly surprising about its contents or
  // the constructor that follows it.
  template <int dim>
  class BiharmonicProblem
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    BiharmonicProblem(const unsigned int fe_degree);

    void run();
    void vmult(VectorType &dst, const VectorType &src) const;

  private:
    void make_grid();
    void setup_system();
    void compute_rhs(VectorType &dst) const;
    void solve();
    void compute_errors();
    void output_results(const unsigned int iteration) const;

    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;

    MappingQ<dim> mapping;

    FE_Q<dim>                 fe;
    DoFHandler<dim>           dof_handler;
    AffineConstraints<double> constraints;

    VectorType solution;
    VectorType system_rhs;

    MatrixFree<dim> matrix_free;

    AlignedVector<VectorizedArray<double>> gamma_over_h;
  };



  template <int dim>
  BiharmonicProblem<dim>::BiharmonicProblem(const unsigned int fe_degree)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , triangulation(MPI_COMM_WORLD)
    , mapping(1)
    , fe(fe_degree)
    , dof_handler(triangulation)
  {}



  // Next up are the functions that create the initial mesh (a once refined
  // unit square) and set up the constraints, vectors, and matrices on
  // each mesh. Again, both of these are essentially unchanged from many
  // previous tutorial programs.
  template <int dim>
  void BiharmonicProblem<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(1);

    pcout << "Number of active cells: " << triangulation.n_active_cells()
          << std::endl
          << "Total number of cells: " << triangulation.n_cells() << std::endl;
  }



  template <int dim>
  void BiharmonicProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             ExactSolution::Solution<dim>(),
                                             constraints);
    constraints.close();

    typename MatrixFree<dim>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      update_quadrature_points | update_values;
    additional_data.mapping_update_flags_inner_faces = update_default;
    additional_data.mapping_update_flags_boundary_faces =
      update_quadrature_points | update_gradients | update_hessians;
    matrix_free.reinit(mapping,
                       dof_handler,
                       constraints,
                       QGauss<1>(fe.degree + 1),
                       additional_data);

    matrix_free.initialize_dof_vector(solution);
    matrix_free.initialize_dof_vector(system_rhs);

    gamma_over_h.resize(matrix_free.n_inner_face_batches() +
                        matrix_free.n_boundary_face_batches());

    for (unsigned int face = 0; face < matrix_free.n_inner_face_batches();
         ++face)
      for (unsigned int v = 0;
           v < matrix_free.n_active_entries_per_face_batch(face);
           ++v)
        {
          const auto p     = fe.degree;
          const auto cell  = matrix_free.get_face_iterator(face, v, true);
          const auto ncell = matrix_free.get_face_iterator(face, v, false);

          gamma_over_h[face][v] = std::max(
            (1.0 * p * (p + 1) /
             cell.first->extent_in_direction(
               GeometryInfo<dim>::unit_normal_direction[cell.second])),
            (1.0 * p * (p + 1) /
             ncell.first->extent_in_direction(
               GeometryInfo<dim>::unit_normal_direction[ncell.second])));
        }

    for (unsigned int face = matrix_free.n_inner_face_batches();
         face < matrix_free.n_inner_face_batches() +
                  matrix_free.n_boundary_face_batches();
         ++face)
      for (unsigned int v = 0;
           v < matrix_free.n_active_entries_per_face_batch(face);
           ++v)
        {
          const auto p    = fe.degree;
          const auto cell = matrix_free.get_face_iterator(face, v);

          gamma_over_h[face][v] =
            (1.0 * p * (p + 1) /
             cell.first->extent_in_direction(
               GeometryInfo<dim>::unit_normal_direction[cell.second]));
        }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void submit_normal_hessian(FEFaceEvaluation<dim,
                                              fe_degree,
                                              n_q_points_1d,
                                              n_components,
                                              Number,
                                              VectorizedArrayType> &phi,
                             const VectorizedArrayType &            value,
                             unsigned int                           q)
  {
    phi.submit_hessian(value * outer_product(phi.get_normal_vector(q),
                                             phi.get_normal_vector(q)),
                       q);
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  VectorizedArrayType
  get_normal_hessian(FEFaceEvaluation<dim,
                                      fe_degree,
                                      n_q_points_1d,
                                      n_components,
                                      Number,
                                      VectorizedArrayType> &phi,
                     unsigned int                           q)
  {
    return phi.get_normal_vector(q) * phi.get_hessian(q) *
           phi.get_normal_vector(q);
  }



  template <int dim>
  void BiharmonicProblem<dim>::vmult(VectorType &      dst,
                                     const VectorType &src) const
  {
    dst = 0.0;

    matrix_free.template loop<VectorType, VectorType>(
      [](const auto &data, auto &dst, const auto &src, const auto cell_range) {
        FEEvaluation<dim, -1, 0, 1, double> phi(data);

        for (auto cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);

            phi.gather_evaluate(src, EvaluationFlags::hessians);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_hessian(phi.get_hessian(q), q);

            phi.integrate_scatter(EvaluationFlags::hessians, dst);
          }
      },
      [&](const auto &data, auto &dst, const auto &src, const auto face_range) {
        FEFaceEvaluation<dim, -1, 0, 1, double> phi_m(data, true);
        FEFaceEvaluation<dim, -1, 0, 1, double> phi_p(data, false);

        for (auto face = face_range.first; face < face_range.second; ++face)
          {
            phi_m.reinit(face);
            phi_p.reinit(face);

            phi_m.gather_evaluate(src,
                                  EvaluationFlags::gradients |
                                    EvaluationFlags::hessians);
            phi_p.gather_evaluate(src,
                                  EvaluationFlags::gradients |
                                    EvaluationFlags::hessians);

            for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
              {
                const auto jmp_normal_derivative =
                  ((phi_m.get_normal_derivative(q) -
                    phi_p.get_normal_derivative(q)));

                const auto avg_normal_hessian =
                  0.5 *
                  (get_normal_hessian(phi_m, q) + get_normal_hessian(phi_p, q));

                submit_normal_hessian(phi_m, -0.5 * jmp_normal_derivative, q);
                phi_m.submit_normal_derivative(-avg_normal_hessian +
                                                 gamma_over_h[face] *
                                                   jmp_normal_derivative,
                                               q);

                submit_normal_hessian(phi_p, -0.5 * jmp_normal_derivative, q);
                phi_p.submit_normal_derivative(+avg_normal_hessian -
                                                 gamma_over_h[face] *
                                                   jmp_normal_derivative,
                                               q);
              }

            phi_m.integrate_scatter(EvaluationFlags::gradients |
                                      EvaluationFlags::hessians,
                                    dst);
            phi_p.integrate_scatter(EvaluationFlags::gradients |
                                      EvaluationFlags::hessians,
                                    dst);
          }
      },
      [&](const auto &data, auto &dst, const auto &src, const auto face_range) {
        FEFaceEvaluation<dim, -1, 0, 1, double> phi(data);

        for (auto face = face_range.first; face < face_range.second; ++face)
          {
            phi.reinit(face);
            phi.gather_evaluate(src,
                                EvaluationFlags::gradients |
                                  EvaluationFlags::hessians);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const auto normal_derivative = phi.get_normal_derivative(q);
                const auto normal_hessian    = get_normal_hessian(phi, q);

                submit_normal_hessian(phi, -normal_derivative, q);

                phi.submit_normal_derivative(
                  -normal_hessian + gamma_over_h[face] * normal_derivative, q);
              }

            phi.integrate_scatter(EvaluationFlags::gradients |
                                    EvaluationFlags::hessians,
                                  dst);
          }
      },
      dst,
      src);
  }



  template <int dim>
  void BiharmonicProblem<dim>::compute_rhs(VectorType &dst) const
  {
    dst = 0.0;

    VectorType dummy;

    const ExactSolution::RightHandSide<dim> right_hand_side;
    const ExactSolution::Solution<dim>      exact_solution;

    matrix_free.template loop<VectorType, VectorType>(
      [&](const auto &data, auto &dst, const auto &, const auto cell_range) {
        FEEvaluation<dim, -1, 0, 1, double> phi(data);

        for (auto cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const auto              p_vect  = phi.quadrature_point(q);
                VectorizedArray<double> f_value = 0.0;
                for (unsigned int v = 0; v < VectorizedArray<double>::size();
                     ++v)
                  {
                    Point<dim> p;
                    for (unsigned int d = 0; d < dim; ++d)
                      p[d] = p_vect[d][v];
                    f_value[v] = right_hand_side.value(p);
                  }

                phi.submit_value(f_value, q);
              }

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      [](const auto &, auto &, const auto &, const auto) {},
      [&](const auto &data, auto &dst, const auto &, const auto face_range) {
        FEFaceEvaluation<dim, -1, 0, 1, double> phi(data);

        for (auto face = face_range.first; face < face_range.second; ++face)
          {
            phi.reinit(face);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const auto p_vect = phi.quadrature_point(q);
                Tensor<1, dim, VectorizedArray<double>> f_grad;
                for (unsigned int v = 0; v < VectorizedArray<double>::size();
                     ++v)
                  {
                    Point<dim> p;
                    for (unsigned int d = 0; d < dim; ++d)
                      p[d] = p_vect[d][v];

                    const auto temp = exact_solution.gradient(p);

                    for (unsigned int d = 0; d < dim; ++d)
                      f_grad[d][v] = temp[d];
                  }

                submit_normal_hessian(phi,
                                      -f_grad * phi.get_normal_vector(q),
                                      q);
                phi.submit_normal_derivative(gamma_over_h[face] * f_grad *
                                               phi.get_normal_vector(q),
                                             q);
              }

            phi.integrate_scatter(EvaluationFlags::gradients |
                                    EvaluationFlags::hessians,
                                  dst);
          }
      },
      dst,
      dummy);
  }



  // @sect4{Solving the linear system and postprocessing}
  //
  // The show is essentially over at this point: The remaining functions are
  // not overly interesting or novel. The first one simply uses a direct
  // solver to solve the linear system (see also step-29):
  template <int dim>
  void BiharmonicProblem<dim>::solve()
  {
    pcout << "   Solving system..." << std::endl;

    compute_rhs(system_rhs);

    ReductionControl     reduction_cotrol(100, 1e-10, 1e-10);
    SolverCG<VectorType> solver(reduction_cotrol);

    // TODO: need a better preconditioner
    solver.solve(*this, solution, system_rhs, PreconditionIdentity());

    constraints.distribute(solution);
  }



  // The next function evaluates the error between the computed solution
  // and the exact solution (which is known here because we have chosen
  // the right hand side and boundary values in a way so that we know
  // the corresponding solution). In the first two code blocks below,
  // we compute the error in the $L_2$ norm and the $H^1$ semi-norm.
  template <int dim>
  void BiharmonicProblem<dim>::compute_errors()
  {
    solution.update_ghost_values();
    {
      Vector<float> norm_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution::Solution<dim>(),
                                        norm_per_cell,
                                        QGauss<dim>(fe.degree + 2),
                                        VectorTools::L2_norm);
      const double error_norm =
        VectorTools::compute_global_error(triangulation,
                                          norm_per_cell,
                                          VectorTools::L2_norm);
      pcout << "   Error in the L2 norm           :     " << error_norm
            << std::endl;
    }

    {
      Vector<float> norm_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution::Solution<dim>(),
                                        norm_per_cell,
                                        QGauss<dim>(fe.degree + 2),
                                        VectorTools::H1_seminorm);
      const double error_norm =
        VectorTools::compute_global_error(triangulation,
                                          norm_per_cell,
                                          VectorTools::H1_seminorm);
      pcout << "   Error in the H1 seminorm       : " << error_norm
            << std::endl;
    }

    // Now also compute an approximation to the $H^2$ seminorm error. The actual
    // $H^2$ seminorm would require us to integrate second derivatives of the
    // solution $u_h$, but given the Lagrange shape functions we use, $u_h$ of
    // course has kinks at the interfaces between cells, and consequently second
    // derivatives are singular at interfaces. As a consequence, we really only
    // integrate over the interior of cells and ignore the interface
    // contributions. This is *not* an equivalent norm to the energy norm for
    // the problem, but still gives us an idea of how fast the error converges.
    //
    // We note that one could address this issue by defining a norm that
    // is equivalent to the energy norm. This would involve adding up not
    // only the integrals over cell interiors as we do below, but also adding
    // penalty terms for the jump of the derivative of $u_h$ across interfaces,
    // with an appropriate scaling of the two kinds of terms. We will leave
    // this for later work.
    {
      const QGauss<dim>            quadrature_formula(fe.degree + 2);
      ExactSolution::Solution<dim> exact_solution;
      Vector<double> error_per_cell(triangulation.n_active_cells());

      FEValues<dim> fe_values(mapping,
                              fe,
                              quadrature_formula,
                              update_values | update_hessians |
                                update_quadrature_points | update_JxW_values);

      FEValuesExtractors::Scalar scalar(0);
      const unsigned int         n_q_points = quadrature_formula.size();

      std::vector<SymmetricTensor<2, dim>> exact_hessians(n_q_points);
      std::vector<Tensor<2, dim>>          hessians(n_q_points);
      for (auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[scalar].get_function_hessians(solution, hessians);
            exact_solution.hessian_list(fe_values.get_quadrature_points(),
                                        exact_hessians);

            double local_error = 0;
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                local_error +=
                  ((exact_hessians[q_point] - hessians[q_point]).norm_square() *
                   fe_values.JxW(q_point));
              }
            error_per_cell[cell->active_cell_index()] = std::sqrt(local_error);
          }

      const double error_norm =
        VectorTools::compute_global_error(triangulation,
                                          error_per_cell,
                                          VectorTools::L2_norm);
      pcout << "   Error in the broken H2 seminorm: " << error_norm
            << std::endl;
    }
    solution.zero_out_ghost_values();
  }



  // Equally uninteresting is the function that generates graphical output.
  // It looks exactly like the one in step-6, for example.
  template <int dim>
  void
  BiharmonicProblem<dim>::output_results(const unsigned int iteration) const
  {
    pcout << "   Writing graphical output..." << std::endl;

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    const std::string filename =
      ("output_" + Utilities::int_to_string(iteration, 6) + ".vtu");
    std::ofstream output_vtu(filename);
    data_out.write_vtu(output_vtu);
  }



  // The same is true for the `run()` function: Just like in previous
  // programs.
  template <int dim>
  void BiharmonicProblem<dim>::run()
  {
    make_grid();

    const unsigned int n_cycles = 2;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << " of " << n_cycles << std::endl;

        triangulation.refine_global(1);
        setup_system();

        solve();

        output_results(cycle);

        compute_errors();
        pcout << std::endl;
      }
  }
} // namespace Step47



// @sect3{The main() function}
//
// Finally for the `main()` function. There is, again, not very much to see
// here: It looks like the ones in previous tutorial programs. There
// is a variable that allows selecting the polynomial degree of the element
// we want to use for solving the equation. Because the C0IP formulation
// we use requires the element degree to be at least two, we check with
// an assertion that whatever one sets for the polynomial degree actually
// makes sense.
int main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace Step47;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const unsigned int fe_degree = 2;
      Assert(fe_degree >= 2,
             ExcMessage("The C0IP formulation for the biharmonic problem "
                        "only works if one uses elements of polynomial "
                        "degree at least 2."));

      BiharmonicProblem<2> biharmonic_problem(fe_degree);
      biharmonic_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
