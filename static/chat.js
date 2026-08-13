
const chatScroll = document.getElementById("chat-scroll");
const emptyState = document.getElementById("empty-state");
const docList = document.getElementById("doc-list");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

function renderDocItem(filename){
    const li = document.createElement("li");
    li.className = "doc-item active hide-scrollbar";
    li.innerHTML = `<span class="dot"></span><span>${filename}</span>`;
    docList.appendChild(li);
}

async function loadDocuments(){
    const response = await fetch('/documents',{
        method: "GET"
    });
    const docs = (await response.json()).documents;
    console.log(docs);
    for (doc of docs){
        renderDocItem(doc.filename);
    }
}

uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    uploadStatus.textContent = "reading pdf...";

    const formData = new FormData();
    formData.append("file", file);
    try{
        const response = await fetch(`/upload`, {
            method: "POST",
            body: formData
        });
        if (!response.ok) throw new Error(`upload failed (${response.status})`);
        const data = await response.json();
        if (!data.exists){
            docList.querySelectorAll(".doc-item").forEach(el => el.classList.remove("active"));
            renderDocItem(data.filename)
            uploadStatus.textContent = "ready";
        }else{
            uploadStatus.textContent = "already existing";
        }
        
        setTimeout(() => (uploadStatus.textContent=""), 2000);

        emptyState.style.display = "none";
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();
    }catch (err){
        uploadStatus.textContent = "upload failed";
        console.error(err);
    }

    fileInput.value = "";
});

function addMessage(role, text){
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    chatScroll.appendChild(div);
    chatScroll.scrollTop = chatScroll.scrollHeight;
    return div;
}

async function sendQuestion(){
    const question = questionInput.value.trim();
    if (!question) return;

    questionInput.value="";
    question.disabled = true;
    sendBtn.disabled = true;

    addMessage("user", question);

    const assistantE1 = document.createElement("div");
    assistantE1.className = "msg assistant";
    assistantE1.innerHTML = `<span class="cursor"></span>`;
    chatScroll.appendChild(assistantE1);
    chatScroll.scrollTop = chatScroll.scrollHeight;

    try{
        const response = await fetch(`/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })           
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        while (true){
            const {done, value} = await reader.read();
            if (done) break;
            fullText += decoder.decode(value, {stream: true});
            assistantE1.innerHTML = `${fullText}<span class="cursor"></span>`;
            chatScroll.scrollTop = chatScroll.scrollHeight;
        }

        assistantE1.textContent = fullText;
    }catch (err){
        assistantE1.textContent = "something went wrong. try again.";
        console.error(err)
    }finally{
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

sendBtn.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuestion();
});

loadDocuments();