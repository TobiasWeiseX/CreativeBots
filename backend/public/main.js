
"use strict";




//idea: generate proxy opject via openapi.json   api(url).login_now()

function API(jwt){

    const handler = {
        get(obj, prop) {
            //return prop in obj ? obj[prop] : 37;

            //alert(prop);

            (async function f(){


            const response = await fetch("/openapi/openapi.json", {
                method: "GET",
                headers: {
                    'accept': '*/*'
                }
            });

            let s = await response.json();

            console.log(s);



            })();



        },
    };

    return new Proxy({}, handler);
}

//API().abc;


async function login(email, pwd){
    const formData = new FormData();
    formData.append("email", email);
    formData.append("password", pwd);
    const response = await fetch("/user/login", {
        method: "POST",
        headers: {
            'accept': '*/*'
        },
        body: formData
    });
    return response.json();
}

async function register(email, pwd){
    const formData = new FormData();
    formData.append("email", email);
    formData.append("password", pwd);
    const response = await fetch("/user/register", {
        method: "POST",
        headers: {
            'accept': '*/*'
        },
        body: formData
    });
    return response.json();
}




async function get_bots(jwt){
    if(jwt){
        const response = await fetch("/bot", {
            method: "GET",
            headers: {
                'accept': '*/*',
                'Authorization': 'Bearer ' + jwt
            }
        });
        return response.json();
    }
    else{
        const response = await fetch("/bot", {
            method: "GET",
            headers: {
                'accept': '*/*'
            }
        });
        return response.json();
    }
}

async function text2speech(txt){
    const formData = new FormData();
    formData.append("text", txt);
    const response = await fetch("/text2speech", {
        method: "POST",
        headers: {
            'accept': '*/*'//,
            //'Authorization': 'Bearer ' + jwt
        },
        body: formData
    });
    return response.json();
}

async function create_bot(jwt, name, visibility, description, llm, sys_prompt){
    const formData = new FormData();
    formData.append("name", name);
    formData.append("visibility", visibility);
    formData.append("description", description);
    formData.append("llm_model", llm);
    formData.append("system_prompt", sys_prompt);

    const response = await fetch("/bot", {
        method: "POST",
        headers: {
            'accept': '*/*',
            'Authorization': 'Bearer ' + jwt
        },
        body: formData
    });
    return response.json();
}

async function* ask_question(bot_id, question, system_prompt=""){
    let socket;
    let room = null;
    //let evt_listener = null;
    let dom_ele = document.head;
    const evt_name = "tokenstream";
    try{
        socket = io();
        socket.on('backend response', data =>{
            console.log(data);
            if(data.room){
                room = data.room;
                socket.off('backend response');
            }
        });

        let done = false;
        //let last_timestamp = null;

        function f(){
            return new Promise((resolve,reject)=>{
                let evt_listener = evt => {
                    //if(evt.timeStamp !== last_timestamp){
                    //last_timestamp = evt.timeStamp;
                    dom_ele.removeEventListener(evt_name, evt_listener);
                    resolve(evt.detail);
                    //}
                    //last_timestamp = evt.timeStamp;
                };
                dom_ele.addEventListener(evt_name, evt_listener);
            });
        }

        socket.on('backend token', obj =>{
            if(!obj.done){
                dom_ele.dispatchEvent(new CustomEvent(evt_name, { detail: obj.data }));
            }
            else{
                done = true;
                socket.off('backend token');
                dom_ele.dispatchEvent(new CustomEvent(evt_name, { detail: "" }));
            }
        });
        socket.emit('client message', {question, system_prompt, bot_id, room});
        while(!done){
            yield f();
        }
        return;
    }
    catch(e){
        console.error(e);
        return;
    }
    finally{
        socket.emit('end');
        socket.close();
        return;
    }
}


function parse_html(html){
    const parser = new DOMParser();
    return parser.parseFromString(html, 'text/html').documentElement;
}

function parse_xml(xml){
    const parser = new DOMParser();
    return parser.parseFromString(xml, 'text/xml').documentElement;
}


function parse_dot_lang(txt){
    let layout = "dot";
    return Viz(txt, {engine:layout});
}


function alert_bot_creation(success){
    let msg, s;
    if(success){
        s = "success";
        msg = "<strong>Success!</strong> Bot created!";
    }
    else{
        s = "danger";
        msg = "<strong>Couldn't create bot!</strong> Something killed that bot!";
    }
    document.getElementById("alert_spawn").innerHTML = `
    <div class="alert alert-${s} alert-dismissible fade show">
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        ${msg}
    </div>
    `;
}

function alert_bot_change(success){
    let msg, s;
    if(success){
        s = "success";
        msg = "<strong>Success!</strong> Bot changed!";
    }
    else{
        s = "danger";
        msg = "<strong>Couldn't change bot!</strong> Something froze that bot!";
    }
    document.getElementById("change_bot_alert_spawn").innerHTML = `
    <div class="alert alert-${s} alert-dismissible fade show">
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        ${msg}
    </div>
    `;
}

window.onload = async ()=>{
    document.documentElement.style.setProperty("--bs-primary-rgb", "45, 124, 172");

    //chat
    let user_input = document.getElementById("user_input");
    let log = document.getElementById("log");
    let submit_btn = document.getElementById("submit_btn");
    let scroll_div = document.getElementById("scroll_div");

    //settings
    let system_prompt = document.getElementById("system_prompt");

    let bot_select = document.getElementById("bot_select");
    let view_select = document.getElementById("view_select");
    let login_btn = document.getElementById("login_btn");
    let logout_btn = document.getElementById("logout_btn");

    let submit_login_btn = document.getElementById("submit_login_btn");
    let submit_register_btn = document.getElementById("submit_register_btn");


    //create bot form
    let create_bot_btn = document.getElementById("create_bot_btn");
    let bot_name = document.getElementById("bot_name");
    let bot_visibility_select = document.getElementById("bot_visibility_select");
    let bot_description = document.getElementById("bot_description");
    let bot_llm_select = document.getElementById("bot_llm_select");
    let bot_system_prompt = document.getElementById("bot_system_prompt");


    //change bot form
    let change_bot_btn = document.getElementById("change_bot_btn");
    let change_bot_name = document.getElementById("change_bot_name");
    let change_bot_visibility_select = document.getElementById("change_bot_visibility_select");
    let change_bot_description = document.getElementById("change_bot_description");
    let change_bot_llm_select = document.getElementById("change_bot_llm_select");
    let change_bot_system_prompt = document.getElementById("change_bot_system_prompt");




    let answer_count = 0;

    function log_msg(nick, msg){
        console.log(nick + ": " + msg);
        log.innerHTML += "<tr><td><b>" + nick + "</b>:</td><td>" + msg + "</td></tr>";
    }

    function scroll_down(){
        scroll_div.scrollTop = scroll_div.scrollHeight;
    }

    function get_bot_name(){
        let i = bot_select.selectedIndex;
        if(i===-1) return "Bot";
        return bot_select.options[i].text;
    }

    function set_bot_list(ls){
        if(ls.length === 0){
            console.error("No bots found!");
        }
        else{
            bot_select.innerHTML = ls.map(bot => `<option value="${bot.id}">${bot.name}</option>`).join("");
        }
    }

    function clean_bot_create_form(){
        bot_name.value = "";
        bot_description.value = "";
        bot_system_prompt.value = "";
    }

    function set_ui_loggedin(b){
        if(b){
            console.log("User logged in!");

            //enable create bot button
            create_bot_btn.removeAttribute("disabled");
            change_bot_btn.removeAttribute("disabled");
            login_btn.style.display = "none";
            logout_btn.style.display = "block";

            document.getElementById("tab2").classList.remove('disabled');
            document.getElementById("tab3").classList.remove('disabled');
        }
        else{
            console.log("User not logged in!");

            //disable create bot button
            create_bot_btn.setAttribute("disabled", "disabled");
            change_bot_btn.setAttribute("disabled", "disabled");
            logout_btn.style.display = "none";
            login_btn.style.display = "block";

            document.getElementById("tab2").classList.add('disabled');
            document.getElementById("tab3").classList.add('disabled');
        }
    }


    //init: are we logged in on start?
    let jwt = localStorage.getItem("jwt");
    if(jwt === null){
        let ls = await get_bots();
        set_bot_list(ls);
        set_ui_loggedin(false);
    }
    else{
        let ls = await get_bots(jwt);
        set_bot_list(ls);
        set_ui_loggedin(true);
    }

    //init chat
    log_msg(get_bot_name(), "Ask a question!");

    //-----init buttons------------
    create_bot_btn.onclick = async ()=>{
        let jwt = localStorage.getItem("jwt");
        if(jwt){
            let name = bot_name.value;
            let visibility = bot_visibility_select.value;
            let description = bot_description.value;
            let llm = bot_llm_select.value;
            let sys_prompt = bot_system_prompt.value;

            if(!name){
                bot_name.focus();
                return;
            }

            if(!sys_prompt){
                bot_system_prompt.focus();
                return;
            }

            try{
                let {bot_id} = await create_bot(jwt, name, visibility, description, llm, sys_prompt);
                alert_bot_creation(true);
                clean_bot_create_form();

                //update bot list
                let ls = await get_bots(jwt);
                set_bot_list(ls);

            }
            catch(err){
                console.error(err);
                console.error("Couldn't create bot!");
                alert_bot_creation(false);
            }
        }
    };



    change_bot_btn.onclick = async ()=>{
        let jwt = localStorage.getItem("jwt");
        if(jwt){
            let name = change_bot_name.value;
            let visibility = change_bot_visibility_select.value;
            let description = change_bot_description.value;
            let llm = change_bot_llm_select.value;
            let sys_prompt = change_bot_system_prompt.value;

            if(!name){
                change_bot_name.focus();
                return;
            }

            if(!sys_prompt){
                change_bot_system_prompt.focus();
                return;
            }

            try{
                let {bot_id} = await change_bot(jwt, name, visibility, description, llm, sys_prompt);
                alert_bot_change(true);
                //clean_bot_create_form();

                //update bot list
                let ls = await get_bots(jwt);
                set_bot_list(ls);

            }
            catch(err){
                console.error(err);
                console.error("Couldn't change bot!");
                alert_bot_change(false);
            }
        }
    };





    submit_login_btn.onclick = async ()=>{
        let nick_ele = document.getElementById("email");
        let pwd_ele = document.getElementById("pass");

        if(!nick_ele.value){
            nick_ele.focus();
            return;
        }

        if(!pwd_ele.value){
            pwd_ele.focus();
            return;
        }

        let nick = nick_ele.value;
        let pwd = pwd_ele.value;


        try{
            let{jwt} = await login(nick, pwd);

            if(!jwt) throw Error("No JWT!");

            localStorage.setItem("jwt", jwt);
            set_ui_loggedin(true);

            let ls = await get_bots(jwt);
            set_bot_list(ls);

            let myModalEl = document.querySelector('#myModal');
            let myModal = bootstrap.Modal.getOrCreateInstance(myModalEl);
            myModal.hide();
        }
        catch(e){
            console.error("Login failed!");
        }
    };




    submit_register_btn.onclick = async ()=>{
        let nick_ele = document.getElementById("register_email");
        let pwd_ele = document.getElementById("register_password");

        if(!nick_ele.value){
            nick_ele.focus();
            return;
        }

        if(!pwd_ele.value){
            pwd_ele.focus();
            return;
        }

        let nick = nick_ele.value;
        let pwd = pwd_ele.value;


        try{
            let{status} = await register(nick, pwd);

            if(status === "success"){
                let myModalEl = document.querySelector('#registerModal');
                let myModal = bootstrap.Modal.getOrCreateInstance(myModalEl);
                myModal.hide();
                alert("Please check your email! And verify your account!");
            }
            else{

                alert("Registration error!");
            }

        }
        catch(e){
            console.error("Registration failed!");
        }
    };









    logout_btn.onclick = async ()=>{
        localStorage.removeItem("jwt");

        set_ui_loggedin(false);

        let ls = await get_bots();
        set_bot_list(ls);
    };

    function replace_dom_code(f, root_ele){
        let eles = root_ele.getElementsByTagName("code");
        for(let i=0; i<eles.length; i++){
            let ele = eles[i];

            //let ele2 = parse_html(f(ele));
            let ele2 = f(ele);

            if(ele2){
                ele.parentNode.replaceChild(ele2, ele);
            }
        }
        return root_ele;
    }



    function is_graph(obj){
        if("nodes" in obj){
            if("edges" in obj || "links" in obj){
                return true;
            }
        }
        return false;
    }


    function rename_attr(obj, old, s){
        if(obj[old]){
            obj[s] = obj[old];
            delete obj[old];
        }
        return obj;
    }



    function translate_graph(obj){
        let ret_obj = {
            nodes: [],
            edges: []
        };

        obj = rename_attr(obj, "links", "edges");

        if(obj.nodes){
            if(Array.isArray(obj.nodes)){
                for(let node of obj.nodes){

                    if(typeof node === "object"){
                        if(node.id){
                            node = rename_attr(node, "name", "label");
                            ret_obj.nodes.push([node.id, { "radius": 15, "color": "orange"} ]);
                        }
                    }

                }
            }
        }

        if(obj.edges){
            if(Array.isArray(obj.edges)){
                for(let edge of obj.edges){
                    if(typeof edge === "object"){
                        edge = rename_attr(edge, "source", "from");
                        edge = rename_attr(edge, "target", "to");
                        if(edge.from){
                            if(edge.to){
                                let e = [edge.from, edge.to, {"color": "black"}];
                                ret_obj.edges.push(e);
                            }
                        }
                    }

                }
            }
        }

        console.log(ret_obj);
        return ret_obj;
    }


    function replace_code(code_ele){
        let txt = code_ele.innerHTML;

        try{
            return parse_html(parse_dot_lang(txt));
        }
        catch(err){
            //console.log(err);
        }

        try{
            let obj = JSON.parse(txt);
            if(is_graph(obj)){
                let s = `<net-graph style="width:400px; height:350px;">${JSON.stringify(translate_graph(obj))}</net-graph>`;
                return parse_html(s);
            }

        }
        catch(err){
            //console.log(err);
        }

        return code_ele;
    }



    submit_btn.onclick = async evt =>{
        let input_string = user_input.value;

        if(input_string.trim() !== ''){
            answer_count += 1;

            user_input.value = "";
            log_msg('User', input_string);
            log.innerHTML += `<tr>
                                <td><b>${get_bot_name()}</b>:</td>
                                <td id="${answer_count}">

                                    <div class="spinner-border"></div>
                                </td>
                              </tr>`;
            let table_cell = document.getElementById(answer_count);

            let acc_text = "";
            for await (let token of ask_question(bot_select.value, input_string, system_prompt.value)){
                //console.log(token);
                acc_text += "" + token;
                switch(view_select.value){
                    case "md":
                        table_cell.innerHTML = "";
                        let ele = replace_dom_code(replace_code, parse_html(marked.parse(acc_text)));
                        table_cell.appendChild(ele);
                        break;

                    case "plain":
                        table_cell.innerHTML = `<pre>${acc_text}</pre>`;
                        break;
                }
                scroll_down();
            }

            /*
            function play() {
              var audio = new Audio('https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3');
              audio.play();
            }
            */


            let final_answer = acc_text;
            console.log(final_answer);


            let extra_s = "";
            let{file} = await text2speech(final_answer);

            //autoplay controls
            extra_s = `
                <audio style="border-radius: 25px;" controls>
                    <source src="${file}" type="audio/mpeg">
                </audio>`;

            //console.log(file);

            switch(view_select.value){

                case "md":
                    table_cell.innerHTML = "";
                    let ele = replace_dom_code(replace_code, parse_html(marked.parse(acc_text) + extra_s));
                    table_cell.appendChild(ele);
                    break;

                case "plain":
                    table_cell.innerHTML = `<pre>${final_answer}</pre>`;
                    break;
            }

            scroll_down();

        }
    };

};















