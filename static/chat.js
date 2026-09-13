
const chatScroll = document.getElementById("chat-scroll");
const emptyState = document.getElementById("empty-state");
const docList = document.getElementById("doc-list");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
var activeDoc;
function renderDocItem(doc_id, filename){
    const li = document.createElement("li");
    li.id = doc_id;
    li.className = "doc-item active hide-scrollbar";
    li.innerHTML = `<span class="dot"></span><span>${filename}</span>`;
    docList.appendChild(li);
    activeDoc = doc_id;
}
docList.addEventListener("click", async (event) => {
    const item = event.target.closest(".doc-item");
    if (!item) return;

    docList.querySelectorAll(".doc-item").forEach(el => el.classList.remove("active"));
    item.classList.add("active");
    emptyState.style.display = "none";
    loadChatHistory(item.id);
    activeDoc = item.id;
});
async function updateChatHistory(role, text){
    data = JSON.stringify({
        role: role,
        text: text
    });
    const response = await fetch(`/chat/${activeDoc}/history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: data
    });
}
async function addMessage(role, text){
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    chatScroll.appendChild(div);
    chatScroll.scrollTop = chatScroll.scrollHeight;
}

async function loadDocuments(){
    const response = await fetch('/documents',{
        method: "GET"
    });
    const docs = (await response.json()).documents;
    for (doc of docs){
        renderDocItem(doc.doc_id, doc.filename);
    }
    docList.querySelectorAll(".doc-item").forEach(el => el.classList.remove("active"));
}
async function loadChatHistory(doc_id){
    questionInput.value="";
    questionInput.disabled = true;
    sendBtn.disabled = true;
    chatScroll.innerHTML = "";
    try{
        const response = await fetch(`/chat/${doc_id}/history`, {
            method: "GET"
        });
        const history = (await response.json()).history;
        for (msg of history){
            addMessage(msg.role, msg.text);
        }
    }catch (err){
        console.log(err);
    }finally{
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();
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
        docList.querySelectorAll(".doc-item").forEach(el => el.classList.remove("active"));
        if (!data.exists){
            renderDocItem(data.doc_id, data.filename);
            uploadStatus.textContent = "ready";

        }else{
            uploadStatus.textContent = "already existing";
            document.getElementById(data.doc_id).classList.add("active");
            activeDoc = data.doc_id;
        }

        
        setTimeout(() => (uploadStatus.textContent=""), 2000);
        emptyState.style.display = "none";
        loadChatHistory(data.doc_id)
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();
    }catch (err){
        uploadStatus.textContent = "upload failed";
        console.error(err);
    }
    console.log(activeDoc);

    fileInput.value = "";
});


async function sendQuestion(){
    const question = questionInput.value.trim();
    if (!question) return;

    questionInput.value="";
    questionInput.disabled = true;
    sendBtn.disabled = true;

    addMessage("user", question);
    const assistantE1 = document.createElement("div");
    assistantE1.className = "msg assistant";
    assistantE1.innerHTML = `<span class="cursor"></span>`;
    chatScroll.appendChild(assistantE1);
    chatScroll.scrollTop = chatScroll.scrollHeight;
    await updateChatHistory("user", question);
    try{
        const response = await fetch(`/chat/${activeDoc}`, {
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
        await updateChatHistory("assistant", fullText);
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