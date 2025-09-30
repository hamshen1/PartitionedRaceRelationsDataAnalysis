# Next Steps

After Anshul sends us the race relations data set and ANCO-HITS algorithm.

We will all use dummy data sets to understand the mechanics of the algorithm before planning out the partitioning, filtering, and analysis.

ANCO-HITS score: This is an algorithm that creates a bipartite graph (look it up) to match users to topics. This is possible because their posts are labeled by an AI to be pro,anti, or neutral. If the user is pro a topic, their node will be closer to the topic node and if they are anti a topic their node will be further from the topic node.

## Partitioning
The proposal is to partition the data into 3 sections and analyze how the ANCO-HITS score changes over time.

## Filtering
Since the avg tweets per user in the data set is <3, in all likely hood we will only get 1-2 tweets per user per partition. But the ANCO-HITS algorithm relies on multiple data points to yield an accurate result. e.g. If the user only has one data point then, all we know is they like/don't like that one thing, but that could represent a wide array of viewpoints. So, Yusuf proposed that we filter the users to only those with 5+ tweets per partition and data within every partition. However, this will greatly decrease our pool of data so we will optimize the cutoff number to get the most valuable dataset.

## Analysis
Finally, we will look for narratives that produced a large change in ANCO-HITS score. 