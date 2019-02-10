Preparation
-----------

### Step 0 - Load all the required Libraries

    library(tidytext)
    library(tidyverse)
    library(tidyr)
    library(tm)

    library(topicmodels)
    library(LDAvis)
    library(slam)
    library(servr)
    library(gridExtra)
    library(ggridges)

### Step 1 - Load the cleaned and processed data

Considering the conciseness, all of the text preprocessing part are done
in `Text_Processing_Modified.rmd` and the data we use here is called
`processed_moments_modified.csv` saved in `output`.

    hm_tbl <- read.csv("../output/processed_moments_modified.csv", stringsAsFactors=FALSE) %>% as.tibble()
    hm_tbl_re <- hm_tbl %>% mutate(age = ifelse(age >= 75, 75, age))

    Corpus <- hm_tbl_re$text %>% VectorSource() %>% VCorpus()
    dtm <- Corpus %>% DocumentTermMatrix()

Happiness V.S. Age
------------------

-   Motivation: Happiness is a kind of emotion of people while the age
    is a character that changes with the time goes by. With the
    increasing of age, the knowledge, status and personality will change
    dynamically.
-   Goal: The main goal of this part is to analysis, how the pattern of
    happiness changes with the increasing of age.

### Question 1 - Demographic bias

    dem_tbl <- hm_tbl %>% count(age, gender, marital, parenthood)
    dem_tbl_re <- dem_tbl %>% mutate(age = ifelse(age >= 75, 75, age))

    dem_tbl %>% 
      group_by(age, gender) %>% 
      summarise(percent = sum(n)) %>% 
      mutate(percent = percent/sum(percent)) %>%
      ggplot(aes(x = age, y = percent, fill = gender)) + 
      geom_col(position = "stack", width=0.8)

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-1-1.png)

    dem_tbl_re %>% 
      group_by(age, gender) %>% 
      summarise(percent = sum(n)) %>% 
      mutate(percent = percent/sum(percent)) %>%
      ggplot(aes(x = age, y = percent, fill = gender)) + 
      geom_col(position = "stack", width=0.87)

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-2-1.png)

    dem_tbl_re %>% 
      group_by(age, marital) %>% 
      summarise(percent = sum(n)) %>% 
      mutate(percent = percent/sum(percent)) %>%
      ggplot(aes(x = age, y = percent, fill = marital)) + 
      geom_col(position = "stack", width=0.8)

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-3-1.png)

    dem_tbl_re %>% 
      group_by(age, parenthood) %>% 
      summarise(percent = sum(n)) %>% 
      mutate(percent = percent/sum(percent)) %>%
      ggplot(aes(x = age, y = percent, fill = parenthood)) + 
      geom_col(position = "stack", width=0.87)

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-4-1.png)

### Question 2 - Single Value Index

    hm_token <- hm_tbl_re %>%
      unnest_tokens(word, text) %>%
      inner_join(get_sentiments("afinn"), by="word") %>%
      mutate(score = abs(score))

    hm_token %>%
      mutate(score = abs(score), age = plyr::round_any(age+5, 10)) %>%
      group_by(age) %>%
      mutate(mean = mean(score)) %>%
      ggplot() + 
      geom_jitter(aes(x = factor(age), y = score), alpha = 0.06) + 
      geom_point(aes(x = factor(age), y = mean), color="red", shape="*", size = 7) +
      labs(x = "Range of Age", 
           y = "Scatter of scores")

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-6-1.png)

    hm_token %>%
      mutate(score = abs(score), age = plyr::round_any(age, 10)) %>%
      ggplot(aes(x = score, y = factor(age))) + 
      geom_density_ridges(aes(x = score, y = factor(age))) + 
      labs(x = "Density of scores", 
           y = "Range of Age")

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-6-2.png)

    hm_tbl_re %>% 
      count(age, predicted_category) %>% 
      group_by(age, predicted_category) %>%
      summarise(percent = sum(n)) %>% 
      mutate(percent = percent/sum(percent)) %>%
      ggplot(aes(x = age, y = percent, fill = predicted_category)) + 
      geom_col(position = "stack", width=0.8)

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    hm_tbl_re %>% 
      count(age, predicted_category) %>% 
      group_by(age, predicted_category) %>%
      summarise(percent = sum(n)) %>% 
      mutate(percent = percent/sum(percent)) %>%
      ggplot(aes(x = age, y = percent, color = predicted_category)) + 
      geom_line()

![](HappyDB_TMining_files/figure-markdown_strict/unnamed-chunk-8-1.png)

    burnin <- 500
    iter <- 1000
    keep <- 30

    k <- 10

    mods <- LDA(dtm, k, 
                method = "Gibbs",
                control = list(burnin = burnin,
                               iter = iter,
                               keep = keep))

    topics <- as.matrix(topics(mods))
    terms <- as.matrix(terms(mods,10))

    # Find required quantities
    phi <- as.matrix(posterior(mods)$terms)
    theta <- as.matrix(posterior(mods)$topics)
    vocab <- colnames(phi)
    term_freq <- col_sums(dtm)

    # Convert to json
    json_lda <- createJSON(phi = phi, 
                           theta = theta,
                           vocab = vocab,
                           doc.length = as.vector(table(dtm$i)),
                           term.frequency = term_freq)

    serVis(json_lda)

    # serVis(json_lda, out.dir = "doc", open.browser = FALSE)

    # serVis(json_lda, out.dir = './', open.browser = FALSE)
    # system("mv index.html results.html")
