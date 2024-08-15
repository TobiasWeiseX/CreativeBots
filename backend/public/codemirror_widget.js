
class CodeMirrorElement extends HTMLElement {

    constructor() {
        super();
        this.attachShadow({mode: 'open'});

        let css_styles = [
          "./lib/codemirror.css",
          "./theme/darcula.css",
          //"./theme/midnight.css"
        ];

        for(let path of css_styles){
          let ele = document.createElement('link');
          ele.setAttribute("rel", "stylesheet");  
          ele.setAttribute("href", path);  
          this.shadowRoot.appendChild(ele);
        }

        let js_scripts = [
          //"lib/codemirror.js",
          //"./mode/htmlmixed/htmlmixed.js",
          //"./mode/css/css.js",
          //"./mode/xml/xml.js",
          //"./mode/clike/clike.js",
          //"./mode/php/php.js"
        ];

        for(let path of js_scripts){
          let ele = document.createElement('script');
          ele.setAttribute("src", path);  
          this.shadowRoot.appendChild(ele);
        }

        this.content_textarea = document.createElement('textarea');
        this.content_textarea.setAttribute("rows", 20);
        this.content_textarea.setAttribute("cols", 150);

        this.shadowRoot.appendChild(this.content_textarea);
        this.slot_ele = document.createElement('slot');
        this.shadowRoot.appendChild(this.slot_ele);
    }

    connectedCallback(){
        let style = this.hasAttribute('style') ? this.getAttribute('style') : "";
        this.content_textarea.style = style;
        let language = this.hasAttribute('language') ? this.getAttribute('language') : "text/x-php";
        let theme = this.hasAttribute('theme') ? this.getAttribute('theme') : "darcula";
        let width = Number(this.hasAttribute('width') ? this.getAttribute('width') : "800");
        let height = Number(this.hasAttribute('height') ? this.getAttribute('height') : "600");

        const editor = CodeMirror.fromTextArea(this.content_textarea, {
          lineNumbers: true,
          matchBrackets: true,
          styleActiveLine: true,
          theme: theme,
          mode: language
        });
        editor.setSize(width, height);
        this.slot_ele.style.display = "none";

        let that = this;
        this.slot_ele.addEventListener('slotchange', e => {
            let source_code = that.innerHTML;

            console.log(source_code);

          /*
            const editor = CodeMirror.fromTextArea(this.content_textarea, {
                lineNumbers: true,
                matchBrackets: true,
                styleActiveLine: true,
                theme: theme,
                mode: language //"text/x-php"
            });


            editor.setSize(width, height);
            */
            
            //let oldCode = editor.getValue();


            editor.setValue(source_code);


            //this.slot_ele.style.display = "none";


            //this.content_textarea.children[0].setAttribute("width", this.content_textarea.style.width);
            //this.content_textarea.children[0].setAttribute("height", this.content_textarea.style.height);
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

customElements.define('code-mirror', CodeMirrorElement);

