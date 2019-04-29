# Titanic Kaggle Competition
# Paul Brendel

setwd("~/Desktop/Kaggle/titanic")

library(tidyverse)
library(janitor)
library(caret)
library(RANN)

# TRAIN

df_train1 <- read_csv("train.csv") %>% clean_names()

# get info from ticket numbers
# separate numbers from letters

df_train_tix <- select(df_train1, passenger_id, ticket) %>% 
  mutate(
    ticket = str_replace_all(ticket, fixed(" "), ""),
    ticket = str_remove_all(ticket, "[:punct:]"),
    ticket_num = as.numeric(str_extract(ticket, "[:digit:]+")),
    ticket_char = tolower(as.character(str_extract(ticket, "[:alpha:]+"))),
    ticket_grp = ifelse(duplicated(ticket_num) | duplicated(ticket_num, fromLast = TRUE),
                        "group", "solo"),
    ticket_char2 = case_when(
      ticket_char == "a" ~ "a",
      ticket_char == "ca" ~ "ca",
      ticket_char == "pc" ~ "pc",
      ticket_char == "sotono" | ticket_char == "stono" | ticket_char == "sotonoq" ~ "st",
      is.na(ticket_char) ~ "na",
      TRUE ~ "other"
      )
  )

df_train2 <- df_train1 %>%
  select(-name, -ticket, -cabin) %>%
  left_join(select(df_train_tix, passenger_id, ticket_grp, ticket_char2), by = "passenger_id") %>% 
  mutate(pclass = as.factor(pclass),
         embarked = as.factor(embarked)) %>% 
  select(-passenger_id)

# impute missing values using KNN; center and scale numerical columns
preproc_values <- preProcess(select(df_train2, -survived), method = c("knnImpute","center","scale"))

df_train3 <- predict(preproc_values, df_train2) 

# train model with k-fold cross validation

set.seed(1234)

fit_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5)

mod_logreg <- train(as.factor(survived) ~ ., data = df_train3, 
                    method = "glm",
                    family = binomial(),
                    trControl = fit_control,
                    na.action = na.pass)

mod_logreg

mod_adaboost <- train(as.factor(survived) ~ ., data = df_train3,
                  method = "adaboost",
                  family = binomial(),
                  trControl = fit_control,
                  na.action = na.pass)
                    
mod_adaboost

# TEST
  
df_test1 <- read_csv("test.csv") %>% clean_names()

df_test_tix <- select(df_test1, passenger_id, ticket) %>% 
  mutate(
    ticket = str_replace_all(ticket, fixed(" "), ""),
    ticket = str_remove_all(ticket, "[:punct:]"),
    ticket_num = as.numeric(str_extract(ticket, "[:digit:]+")),
    ticket_char = tolower(as.character(str_extract(ticket, "[:alpha:]+"))),
    ticket_grp = ifelse(duplicated(ticket_num) | duplicated(ticket_num, fromLast = TRUE),
                        "group", "solo"),
    ticket_char2 = case_when(
      ticket_char == "a" ~ "a",
      ticket_char == "ca" ~ "ca",
      ticket_char == "pc" ~ "pc",
      ticket_char == "sotono" | ticket_char == "stono" | ticket_char == "sotonoq" ~ "st",
      is.na(ticket_char) ~ "na",
      TRUE ~ "other"
    )
  )

df_test2 <- df_test1 %>%
  select(-name, -ticket, -cabin) %>%
  left_join(select(df_test_tix, passenger_id, ticket_grp, ticket_char2), by = "passenger_id") %>% 
  mutate(pclass = as.factor(pclass),
         embarked = as.factor(embarked)) %>% 
  select(-passenger_id)

preproc_values2 <- preProcess(df_test2, method = c("knnImpute","center","scale"))

df_test3 <- predict(preproc_values2, df_test2) 

# use models to make predictions on test data

pred_logreg <- predict.train(mod_logreg, df_test3, type = "raw")
pred_adaboost <- predict.train(mod_adaboost, df_test3, type = "raw")

table(pred_logreg)
table(pred_adaboost)

submission1 <- data_frame("PassengerId" = df_test1$passenger_id, "Survived" = pred_logreg)
submission2 <- data_frame("PassengerId" = df_test1$passenger_id, "Survived" = pred_adaboost)

write_csv(submission1, "pred_logreg.csv")
write_csv(submission2, "pred_adaboost.csv")

