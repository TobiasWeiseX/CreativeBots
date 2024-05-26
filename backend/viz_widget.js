
class DotGraphElement extends HTMLElement {

    constructor() {
        super();
        this.attachShadow({mode: 'open'});
        this.content_div = document.createElement('div');
        this.shadowRoot.appendChild(this.content_div);
        this.slot_ele = document.createElement('slot');
        this.shadowRoot.appendChild(this.slot_ele);
        this.dot_text = "";
    }

    connectedCallback(){
        let layout = this.hasAttribute('layout') ? this.getAttribute('layout') : "fdp";
        let style = this.hasAttribute('style') ? this.getAttribute('style') : "";
        this.content_div.style = style;
        let that = this;
        this.slot_ele.addEventListener('slotchange', e => {
            let graphviz_text = that.textContent;
            this.dot_text = graphviz_text;
            this.content_div.innerHTML = Viz(this.dot_text, {engine:layout});
            this.slot_ele.style.display = "none";
            this.content_div.children[0].setAttribute("width", this.content_div.style.width);
            this.content_div.children[0].setAttribute("height", this.content_div.style.height);
        });
    }

    disconnectedCallback() {

    }

    attributeChangedCallback(name, oldValue, newValue) {
        //this.displayVal.innerText = this.value;
    }

    get layout(){

    }

    set layout(x){

    }

    get value(){
        //dot code
    }

    set value(x){

    }

}

customElements.define('dot-graph', DotGraphElement);

