
"use strict";


class TabsElement extends HTMLElement {

    constructor() {
        super();
        this.attachShadow({mode: 'open'});

        this.style_ele = document.createElement('style');
        this.style_ele.innerHTML = `
        
        .wrapper_div{
            border: 1px solid #c5c5c5;
            padding: 1px;

        }

        .button_div{


        }
        
        .tab_btn{

            color: red;

        }

        `;
        this.shadowRoot.appendChild(this.style_ele);

        this.wrapper_div = document.createElement('div');
        this.wrapper_div.classList.add("wrapper_div");

        this.shadowRoot.appendChild(this.wrapper_div);
        this.wrapper_div.style.height = this.getAttribute('height');

        this.buttonbar = document.createElement('div');
        this.buttonbar.classList.add("button_div");
        this.buttonbar.style.display = "flex";
        this.buttonbar.style["flex-direction"] = "row";
        this.wrapper_div.appendChild(this.buttonbar);

        this.tabs_div = document.createElement('div');
        this.wrapper_div.appendChild(this.tabs_div);

        this.slot_ele = document.createElement('slot');
        this.shadowRoot.appendChild(this.slot_ele);

        this.id_counter = 0;
    }


    clearTabs(){
        this.buttonbar.innerHTML = "";
        this.tabs_div.innerHTML = "";
    }

    addTab(title, content="", active=false){
        let that = this;
        let id = this.id_counter;
        let btn = document.createElement('button');
        btn.classList.add("tab_btn");
        btn.setAttribute("id", "btn_"+id);
        btn.onclick = ()=>{
            that.setActiveTabById(id);
        };
        btn.innerHTML = title;
        this.buttonbar.appendChild(btn);

        let tab_div = document.createElement('div');
        tab_div.setAttribute("id", "tab_div_"+this.id_counter);
        tab_div.style.display = "none";
        tab_div.innerHTML = content;

        this.tabs_div.appendChild(tab_div);
        if(active){
            this.setActiveTabById(id);
        }
        this.id_counter++;
        return this.id_counter - 1;
    }

    setActiveTabById(id){
        for(let i=0; i<this.tabs_div.children.length; i++){
            let tab_div = this.tabs_div.children[i];
            if(tab_div.hasAttribute("id") && tab_div.getAttribute("id") === "tab_div_"+id){
                this.children[i].setAttribute("selected", "selected");
                tab_div.style.display = "block";
            }
            else{
                if(this.children[i].hasAttribute("selected")){
                    this.children[i].removeAttribute("selected");
                }
                tab_div.style.display = "none";
            }
        }
    }

    connectedCallback(){
        this.wrapper_div.style.height = this.getAttribute('height');
        let that = this;
        this.slot_ele.addEventListener('slotchange', e => {
            that.clearTabs();
            for(let i=0; i<that.children.length; i++){
                let ele = that.children[i];
                this.addTab(ele.getAttribute("title"), ele.innerHTML, ele.hasAttribute("selected"));
            }
        });
    }

    disconnectedCallback() {

    }

    attributeChangedCallback(name, oldValue, newValue) {
        if(name === "height"){
            this.wrapper_div.style.height = newValue;
        }
    }

    get height(){
        return this.wrapper_div.style.height;
    }

    set height(x){
        this.wrapper_div.style.height = x;
    }

}

// Neues Element definieren
customElements.define('j-tabs', TabsElement);

class TabElement extends HTMLElement {

    constructor() {
        super();
        this.attachShadow({mode: 'open'});
        this.div = document.createElement('div');
        this.div.style.display = "none";
        this.shadowRoot.appendChild(this.div);
        this.slot_ele = document.createElement('slot');
        this.div.appendChild(this.slot_ele);
    }

    connectedCallback(){

    }

    disconnectedCallback() {

    }

    attributeChangedCallback(name, oldValue, newValue) {

    }

    get title(){

    }

    set title(x){

    }

}

// Neues Element definieren
customElements.define('j-tab', TabElement);

