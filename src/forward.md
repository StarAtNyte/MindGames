  Idea 1: Dynamic Persona Adaptation

  Instead of having a fixed strategy for each role, the agent could dynamically adopt and change its persona based on the game's
  flow. This is about playing the social meta-game, not just the role.

  Concept: The agent actively manages how it's perceived by other players and can change its behavior if its current persona is
  proving ineffective.

  How it would work:

   1. Persona Selection: At the start of the game, the agent uses the LLM to pick a persona from a list like: "The Aggressive Accuser,"
      "The Quiet Analyst," "The Helpful but Naive Friend," "The Scared and Defensive Target."
   2. Persona-driven Actions: All prompts would be modified to include this persona.
       * Prompt: "You are the {Doctor}, and you have adopted the persona of 'The Quiet Analyst'. Based on this persona, what do you say 
         during the discussion?"
   3. The Novelty - Persona Critique & Shift: The agent periodically asks the LLM to evaluate its current persona's effectiveness.
       * Meta-Prompt: "My current persona is 'The Aggressive Accuser'. I am now being targeted by three other players. Is this persona 
         failing? Should I shift to a 'Scared and Defensive' persona to appear less threatening and redirect suspicion? Justify the 
         strategic shift."

  Why it's Novel: This moves the agent from simply following a role-based script to actively performing and managing a character. It's a
   higher level of social strategy, making the agent far less predictable and more human-like.


  Idea 2: Strategic Deception Campaigns

  For the Mafia role, we can elevate simple lying into planned, multi-turn information operations.

  Concept: The agent devises and executes a long-term plan to frame an innocent player or create a specific false narrative.

  How it would work:

   1. Campaign Planning (Mafia Only): At the start of the game or after a key event, the agent uses the LLM to create a plan.
       * Planning Prompt: "As Mafia, devise a three-step plan to frame Player 4 (an innocent) as the Detective. The plan should be 
         subtle and unfold over several turns."
   2. Store and Execute: The LLM might output a plan like:
       1. Turn 2: Publicly agree with a minor, irrelevant point Player 4 makes to create false rapport.
       2. Turn 3: After someone is killed, say 'Player 4 seemed to be onto something, maybe they know more than they let on,' subtly 
          implying they have an investigative role.
       3. Turn 4: If Player 4 makes an accusation, counter by saying 'That's a weak accusation for a supposed Detective,' fully 
          committing to the frame.
   3. The agent stores this plan in its memory and executes each step, tracking its progress.

  Why it's Novel: This gives the agent long-term strategic intent and the ability to execute complex, multi-stage deceptions. It's
  the difference between telling a lie and weaving a web of deceit, which is a hallmark of high-level play in social deduction games.