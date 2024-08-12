
class NetGraphElement extends HTMLElement {

    static get observedAttributes() {
        return ['G'];
    }

    constructor() {
        super();
        this.attachShadow({mode: 'open'});
        this.content_div = document.createElement('div');
        this.shadowRoot.appendChild(this.content_div);
        this.slot_ele = document.createElement('slot');
        this.shadowRoot.appendChild(this.slot_ele);
        this.G = new jsnx.MultiDiGraph();
    }

    connectedCallback(){
        let style = this.hasAttribute('style') ? this.getAttribute('style') : "";
        let weighted = this.hasAttribute('weighted') ? JSON.parse(this.getAttribute('weighted')) : false;
        let withLabels = this.hasAttribute('withLabels') ? JSON.parse(this.getAttribute('withLabels')) : true;
        let label_color = this.hasAttribute('labelColor') ? this.getAttribute('labelColor') : "black";

        this.content_div.style = style;
        let that = this;

        jsnx.draw(that.G, {
            element: that.content_div,
            weighted,
            withLabels,
            labelStyle: {fill: label_color},
            edgeStyle: {
                'stroke-width': 5,
                fill: d => d.data[0].color
            },
            nodeStyle: {
                fill: d => d.data.color
            },
            nodeAttr: {
                r: d => d.data.radius | 10,
                title: d => d.label
            }
        }, true); //true ensures redrawing

        this.slot_ele.addEventListener('slotchange', e => {
            let text = that.innerText.trim();
            let{nodes, edges} = JSON.parse(text);

            for(let[id, data] of nodes){
                that.G.addNode(id, data);
            }

            for(let[a, b, data] of edges){
                that.G.addEdge(a, b, data);
            }

            jsnx.draw(that.G, {
                element: that.content_div,
                weighted,
                withLabels,
                labelStyle: {fill: label_color},
                edgeStyle: {
                    'stroke-width': 5,
                    fill: d => d.data[0].color
                },
                nodeStyle: {
                    fill: d => d.data.color
                },
                nodeAttr: {
                    r: d => d.data.radius | 10,
                    title: d => d.label
                }
            }, true); //true ensures redrawing

            that.slot_ele.style.display = "none";
            that.content_div.children[0].setAttribute("width", that.content_div.style.width);
            that.content_div.children[0].setAttribute("height", that.content_div.style.height);
        });
    }

    disconnectedCallback() {

    }

}

customElements.define('net-graph', NetGraphElement);
