library(tidyr)
library(ggplot2)
library(ggthemes)
library(tibble)
library(dplyr)
library(patchwork)
library(scales)
library(forcats)

theme_set(theme_clean())
# theme_set(theme_gdocs())

data <- read.csv("./multisort_results.csv", header = TRUE)

# data <- gather(threads, ) %>%
# data <- gather(data, variable, value, sddm_time, diag_max_time, create_time, b_min_reduce_time) %>%
# data <- pivot_longer(data, c(sddm_time, diag_max_time, create_time, b_min_reduce_time, b_min_critical_time, b_min_tree_time), names_to = "variable", values_to = "value") %>%

# threads <- gather(data, variable, value, sddm_time, diag_max_time, create_time, b_min_reduce_time) %>%
# data <- gather(data, variable, value, limit) %>%
#     select(variable, value, threads)
# data <- data[data$elements == 10000000, ]

facet_labs <- c(
    
)

data$elements <- factor(data$elements)
data$limit <- factor(data$limit)
data$threads <- factor(data$threads)
# data$grps <- with(data, interaction(limit,  elements, sep = " limit, with elements = "))

# data <- select(data, elements, threads, time)

my_theme <- theme(
    # legend.position = c(.05, .85),
    legend.position = "right",
    legend.key.size = unit(.5, "cm"),
    legend.text = element_text(size = 7)
)
# , labeller = as_labeller(facet_labs)
graph_threads <- ggplot(data = data, aes(x = threads, y = time, color=limit, group=limit)) +
    # facet_grid(elements ~ ., scales="free_y") +
    facet_grid(elements ~ .) +
    # geom_smooth(method = "glm") +
    geom_line() +
    geom_point() +
    scale_y_continuous(trans = "log", labels = scientific) +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
    labs(y = "Time log(s)", x = "Number of Threads", color = "Minimum array size\nbefore falling back\nto sequential sort") +
    # my_theme #+
    # theme(legend.position = c(0.8, 0.3))


pdf("graphs.pdf", width = 10, height = 10)
print(graph_threads)
dev.off()
