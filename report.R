---
title: "R Notebook"
output: html_notebook
---


library(dplyr)

loan_data = read.csv('https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv', header = TRUE)

head(loan_data)

loan_data <-  loan_data |> 
  select(-ZIP.Code)

summary(loan_data)


library("skimr")

skim(loan_data)



library(ggplot2)
library(purrr)
library(tidyr)

ggplot(data = gather(loan_data), aes(x = value, fill = key)) +
  geom_histogram(bins = 10, color = "black") +
  facet_wrap(~key, nrow = 4, ncol = 3, scales = "free") +
  theme_bw()


DataExplorer::plot_bar(loan_data, ncol = 3)

DataExplorer::plot_histogram(loan_data, ncol = 3)

DataExplorer::plot_boxplot(loan_data, by = "Personal.Loan", ncol = 3)


data <- c(99, 1)

labels <- c("which can be predicted", "which cannot be predicted")

colors <- c("red", "green")

pie(data, labels = labels, col = colors)

loan_data$Personal.Loan = as.factor(loan_data$Personal.Loan)

library("data.table")
library("mlr3verse")


set.seed(212) # set seed for reproducibility
loan_task <- TaskClassif$new(id = "PersonalLoan",
                               backend = loan_data, # <- NB: no na.omit() this time
                               target = "Personal.Loan",
                               positive = "1")


set.seed(212)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)
set.seed(212)
bootstrap <- rsmp("bootstrap", ratio=0.8, repeats=3)
bootstrap$instantiate(loan_task)
set.seed(212)
holdout <- rsmp("holdout", ratio=0.7)
holdout$instantiate(loan_task)


set.seed(212)
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
set.seed(212)
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
set.seed(212)
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
set.seed(212)
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
set.seed(212)
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")


set.seed(212)
res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_xgboost,
                    lrn_log_reg,
                    lrn_ranger
                    ),
  resampling = list(cv5)
), store_models = TRUE)
res
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


set.seed(212)
res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_xgboost,
                    lrn_log_reg,
                    lrn_ranger
                    ),
  resampling = list(bootstrap)
), store_models = TRUE)
res
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


set.seed(212)
res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_xgboost,
                    lrn_log_reg,
                    lrn_ranger
                    ),
  resampling = list(holdout)
), store_models = TRUE)
res
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


set.seed(212)
trees <- res$resample_result(5)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
tree1_rpart


library(mlr3tuning)

set.seed(212)

tuning <- ps( 
  mtry = p_int(lower = 1, upper = 10),
  num.trees = p_int( lower = 50, upper = 130),
  max.depth = p_int(lower = 2, upper = 10),
  min.node.size = p_int(lower = 1, upper = 10)
  )

Ranger_instance <- TuningInstanceSingleCrit$new(
  task = loan_task,
  learner = lrn_ranger,
  resampling = holdout,
  measure = msr("classif.acc"),
  search_space = tuning,
  terminator = trm("evals", n_evals = 100)
)
tuning_strategy <- tnr("grid_search", resolution = 10)

tuning_strategy$optimize(Ranger_instance) 


set.seed(118)
lrn_ranger1   <- lrn("classif.ranger", predict_type = "prob",mtry=8, num.trees=58, max.depth=10, min.node.size=4, splitrule = "gini")
set.seed(118)
lrn_ranger2   <- lrn("classif.ranger", predict_type = "prob",mtry=8, num.trees=58, max.depth=10, min.node.size=4, splitrule = "extratrees")
set.seed(118)
lrn_ranger3   <- lrn("classif.ranger", predict_type = "prob",mtry=8, num.trees=58, max.depth=10, min.node.size=4, splitrule = "hellinger")


set.seed(212)
res_ranger1 <- resample(loan_task, lrn_ranger1, cv5, store_models = TRUE)
set.seed(212)
res_ranger2 <- resample(loan_task, lrn_ranger2, bootstrap, store_models = TRUE)
set.seed(212)
res_ranger3 <- resample(loan_task, lrn_ranger3, holdout, store_models = TRUE)



res_ranger1$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

res_ranger2$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

res_ranger3$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


credit_data = read.csv('https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv', header = TRUE)
colnames(credit_data)[9] <- "Status"
credit_data$Status <- ifelse(credit_data$Status == 0, "bad", "good")
library("rsample")
set.seed(212) 
credit_split <- initial_split(credit_data)
credit_train <- training(credit_split)
credit_split2 <- initial_split(testing(credit_split), 0.5)
credit_validate <- training(credit_split2)
credit_test <- testing(credit_split2)


library("recipes")

cake <- recipe(Status ~ ., data = credit_data) %>%
  step_impute_mean(all_numeric()) %>% # impute missings on numeric values with the mean
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = credit_train) # learn all the parameters of preprocessing on the training data

credit_train_final <- bake(cake, new_data = credit_train) # apply preprocessing to training data
credit_validate_final <- bake(cake, new_data = credit_validate) # apply preprocessing to validation data
credit_test_final <- bake(cake, new_data = credit_test) # apply preprocessing to testing data


library("keras")

credit_train_x <- credit_train_final %>%
  select(-starts_with("Status_")) %>%
  as.matrix()
credit_train_y <- credit_train_final %>%
  select(Status_bad) %>%
  as.matrix()

credit_validate_x <- credit_validate_final %>%
  select(-starts_with("Status_")) %>%
  as.matrix()
credit_validate_y <- credit_validate_final %>%
  select(Status_bad) %>%
  as.matrix()

credit_test_x <- credit_test_final %>%
  select(-starts_with("Status_")) %>%
  as.matrix()
credit_test_y <- credit_test_final %>%
  select(Status_bad) %>%
  as.matrix()


deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(credit_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  credit_train_x, credit_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(credit_validate_x, credit_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict(credit_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict(credit_test_x) %>% `>`(0.5) %>% as.integer()


table(pred_test_res, credit_test_y)
yardstick::accuracy_vec(as.factor(credit_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(credit_test_y, levels = c("1","0")),
                       c(pred_test_prob))
