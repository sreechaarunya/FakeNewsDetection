// ===============================
// Elements
// ===============================
const textarea = document.getElementById("news");
const words = document.getElementById("words");
const chars = document.getElementById("chars");
const clearBtn = document.getElementById("clear");
const themeBtn = document.getElementById("theme-btn");

// ===============================
// Word & Character Counter
// ===============================
function updateCounter(){

    const text = textarea.value;

    chars.innerHTML = "Characters : " + text.length;

    const count = text.trim() === ""
        ? 0
        : text.trim().split(/\s+/).length;

    words.innerHTML = "Words : " + count;

}

textarea.addEventListener("input", updateCounter);

// ===============================
// Clear Text
// ===============================
clearBtn.addEventListener("click",()=>{

    textarea.value="";

    updateCounter();

    textarea.focus();

});

// ===============================
// Dark / Light Mode
// ===============================
themeBtn.addEventListener("click",()=>{

    document.body.classList.toggle("light");

    if(document.body.classList.contains("light")){

        themeBtn.innerHTML="☀";

    }

    else{

        themeBtn.innerHTML="🌙";

    }

});

// ===============================
// Loading Animation
// ===============================
document.querySelector("form").addEventListener("submit",function(){

    const btn=document.querySelector(".predict");

    btn.innerHTML="<i class='fa-solid fa-spinner fa-spin'></i> Analyzing...";

    btn.disabled=true;

});

// Initialize
updateCounter();