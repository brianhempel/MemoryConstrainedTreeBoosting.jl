push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

import Random
using MemoryConstrainedTreeBoosting

using Profile

Random.seed!(123456)

# See https://catboost.ai/#benchmark for comparison.
#
#          CatBoost           LightGBM           XGBoost           H20
#          Tuned    Default   Tuned    Default   Tuned   Default   Tuned   Default
# Adult    0.26974  0.27298   0.27602  0.28716   0.27542 0.28009   0.27510 0.27607
#                   +1.21%    +2.33%   +6.46%    +2.11%  +3.84%    +1.99%  +2.35%

# Single thread: 0.2812351

function make_numeric(column)
  # If numeric, pass through. Map ? to 0
  map(column) do val
    val == "?" ? 0f0 : parse(Float32, val)
  end
end

function column_to_numeric_columns_function(column, labels)
  try
    make_numeric(column) # If it fails, we'll catch.
    (column -> make_numeric(column))
  catch
    # If categorical, determine number of categories.
    categories     = unique(column)
    category_count = length(categories)
    if category_count == 2
      # 1 one-hot column
      (column -> Vector{Float32}(column .== categories[1]))
    else
    # elseif category_count == 3
      # One-hot columns
      make_one_hot_columns(column) = begin
        cols =
          map(categories) do category
            Vector{Float32}(column .== category)
          end

        hcat(cols...)
      end
      make_one_hot_columns
    # else
    #   # calculate label prevalence
    #   positives = column[labels .== 1f0]
    #   # slow way, oh well
    #   category_values =
    #     map(categories) do category
    #       count(positives .== category) / Float32(count(column .== category))
    #     end
    #
    #   make_prevalence_column(column) = begin
    #     map(column) do category
    #       cat_i = findfirst(categories .== category)
    #       category_values[cat_i]
    #     end
    #   end
    #   make_prevalence_column
    end
  end
end

# Returns (data, labels)
function load_income_file(file_name; column_functions = nothing)
  raw_data = map(line -> split(line, ", "), split(read(joinpath(@__DIR__, "adult_income_dataset", file_name), String), "\n"))

  raw_columns =
    map(1:15) do col_i
      map(row -> row[col_i], raw_data)
    end

  # Last column is label.
  labels = Vector{Float32}(raw_columns[15] .== ">50K")

  if isnothing(column_functions)
    column_functions =
      map(1:14) do col_i
        column_to_numeric_columns_function(raw_columns[col_i], labels)
      end
  end

  col_chunks =
    map(1:14) do col_i
      column_functions[col_i](raw_columns[col_i])
    end

  # for chunk in col_chunks
  #   println(size(chunk))
  # end

  (hcat(col_chunks...), labels, column_functions)
end

X, y, column_functions                = load_income_file("adult.data")
validation_data, validation_labels, _ = load_income_file("adult.test"; column_functions = column_functions)


bin_splits        = prepare_bin_splits(X)
X_binned          = apply_bins(X, bin_splits)
validation_binned = apply_bins(validation_data, bin_splits)

iteration_callback() =
  make_callback_to_track_validation_loss(
      validation_binned,
      validation_labels;
      max_iterations_without_improvement = 150
    )

train_on_binned(X_binned, y, iteration_count = 2, feature_fraction = 0.9, max_leaves = 6, bagging_temperature = 0.25, iteration_callback = iteration_callback())

@time trees = train_on_binned(
  X_binned, y,
  iteration_count = 10_000,
  learning_rate = 0.01,
  l2_regularization = 10.0,
  feature_fraction = 0.4,
  min_data_weight_in_leaf = 2.9,
  max_delta_score = 5.0,
  max_leaves = 120,
  max_depth = 8,
  bagging_temperature = 0.0,
  iteration_callback = iteration_callback()
)
println("")

ŷ1 = predict(validation_data, bin_splits, trees)

path = tempname(cleanup = true)
save(path, bin_splits, trees)
load_unbinned_predictor(path)
@time save(path, bin_splits, trees)
unbinned_predict = @time load_unbinned_predictor(path)

ŷ2 = unbinned_predict(validation_data)
ŷ2 = @time unbinned_predict(validation_data)

println(size(validation_data))
println(sum(abs.(ŷ2 .- ŷ1)))
@assert sum(abs.(ŷ2 .- ŷ1)) / size(validation_data,1) < 0.00001

bin_splits, trees = load(path)

ŷ3 = predict(validation_data, bin_splits, trees)

println(sum(abs.(ŷ3 .- ŷ1)))
@assert sum(abs.(ŷ3 .- ŷ1)) / size(validation_data,1) < 0.00001