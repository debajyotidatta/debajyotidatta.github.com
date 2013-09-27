---
layout: post
category : ILHAIRE
tagline: "A really brief write-up of ILHAIRE"
tags : [hci, ml , ai, animation]
---
{% include JB/setup %}


## Lip, eyebrow and head animation synthesis

The lip animation synthesis is based on the close relationship between pseudo phonemes of laughter and lip motion. A Gaussian Mixture Model(GMM) is used to learn this relationship. For training the GMM, the lip motion features are clustered by 13 annotated pseudo-phonemes. For the synthesis one pseudo phoneme sequence including the duration of each sequence is used to establish a sequence of Gaussian distribution functions. The determined sequence of Gaussian distribution functions (Tokuda, Yoshimura, Masuko, Kobayashi, & Kitamura, 2000) is used to synthesize directly the smoothed trajectories.
The eyebrow and head animation synthesis is based on choosing the optimal sequences of phoneme trajectories for each of the head rotations: pitch, yaw and roll. This cost function is itself a sum of two other sub cost functions of duration of each phoneme and distances of consecutive phonemes. The final cost function chooses an optimal sequence from the database such that the value of the cost function is the least and thus if there are any differences in durations, between the chosen sequence and the actual sequence, the corresponding generated sequence is stretched or contracted as required.
The above two approaches for “lip” and “head and eyebrow” give the final animation of the head.


##Shoulder and Torso movement generation from breathing

Laughter animation synthesis is incomplete without the synthesis of animation from shoulders and torso. The laughter of the virtual agent will primarily depend on how natural the animation of shoulders and torso is. This is because both shoulder and torso play an extremely critical role in laughter and are primarily characterized by the rapid inhalations and exhalations during laughter which been taken into account, the details of which are described below.
The extent to which the shoulders and torso moves, depends not only on the period of the exhalations and inhalations but also on the magnitude of the laughter intensity. All this is controlled by a PID controller where the transition between different respiration phases directly depends on the parameters, which are “duration” and “desire” in this case. The duration sequences refer to the duration of each phoneme’s exhalation or inhalation cycles. The desire value can be thought of as the magnitude of the exhalation or inhalation cycle due to the corresponding phoneme. A combination of these two desire and duration sequences give the resultant movement for the shoulder and the torso. 
The shoulder and torso thus takes into account the respirations during the utterances of the laughter phonemes so that the output of this is synchronized with the facial animation.


##Future Work

As part of the future work correlations of speech features with different body parts need to be explored and if such correlations do exist the goal would be to synthesize body animation using these features. Based on the results we will use a speech driven approach to drive the animation of the virtual agent for the different body parts of the virtual agent. Since human perception can evaluate the naturalness of laughter animation objective and subjective evaluation needs to be done on the aforementioned approach.
Motion capture data-driven approach is widely used both for high quality and natural ness of animation in a wide array of industries like gaming and animation movies. Exploring this possibility to synthesize the animation of the virtual agent through the motion-graph approach to build natural animation can give promising results for this project. The target would then be to use different input features for animating different parts of the virtual agents’ body so that the entire animation of the virtual agent looks good. For example if breathing has a more profound impact on torso than on head then breathing should be used to drive the torso and laughter phonemes to drive the head. This would essentially involve having an optimized cost function that connects the relative motions to generate a smooth animation for the entire body.
