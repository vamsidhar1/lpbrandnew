function lambda(input, callback) {
    // Set conversation data.
    let conversation = input.payload;
    // Some processing ...
    // Result can be either an object or array.
    // Hint: Make sure to only return each type once.
    let result = [
        {
            type: "systemMessage", // Returns a system message into the conversation.
            text: "since we havent heard from you we will connect you with a live agent"
        },
        {
            type: "transfer", // Transfers the conversation.
            skillId: "1747495870" // Transfer to different skill.
            //agentId: "123456" // Propose an agent.
        },

    ];
    callback(null, result);
}
