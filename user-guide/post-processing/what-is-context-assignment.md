---
description: Explanation and example of context assignment.
icon: question
---

# What is context assignment?

After training SIT-FUSE, the model outputs **clusters** of what SIT-FUSE thinks are similar features (aka patterns). Each cluster corresponds to a number, but this is hard to understand, so we visualize these clusters in a GeoTiff file. Each cluster in this format is a unqiue color to show the patters SIT-FUSE is picking up, rather than just a number.

However, this isn't the final form. We need to do something called context assignment. Context assignment is the process of assigning truth labels to SIT-FUSE's output so we can see SIT-FUSE's prediction on every feature we're looking for. Our truth labels tell the model where exactly the feature is (aka the truth), and then the model uses this information to show us the areas it determined to be that feature.

***

In this wildfire example, there are 3 labels: background, fire, and smoke. Applying these labels to the SIT-FUSE clusters output gives us the final context assigned output on the right.&#x20;

In purple we see what SIT-FUSE classified as smoke, where as if we zoom in, we can see the pink active fire the model picked up. However, we see that SIT-FUSE wasn't sure whether to classify the hazy smoke as smoke, and thus refrained from doing so. Additionally, background is used as a "negative" label so that the model can properly differentiate between the background and any other class. But we don't necessarily need the final output from the background, hence why it isn't included in the final output.

<figure><img src="../../.gitbook/assets/Screenshot 2025-07-23 at 11.10.43 AM.png" alt=""><figcaption></figcaption></figure>

We see that despite having very rough labels, i.e. jagged, un-precise, blocky, we get a very clean output with precise borders aligning very well to those of the original image.&#x20;

***

The next pages will demonstrate how to apply context to various cases such as wildfire imagery or harmful algal bloom (HAB) data.
