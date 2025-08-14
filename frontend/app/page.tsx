'use client'
import Image from "next/image";
import { useState } from "react";

export default function Home() {


    const [color, setColor] = useState("");

    //const handleClick = () => setCount(count + 1);




    function add_geo_location(){
        function callback(position) {
            let s = "My current position is: Latitude: " + position.coords.latitude + " Longitude: " + position.coords.longitude + " !";
            console.log(s);
            document.getElementById("system_prompt").innerHTML += " " + s;
        }
        if(navigator.geolocation){
            navigator.geolocation.getCurrentPosition(callback);
        }
        else {
            console.log("Geolocation is not supported by this browser.");
        }
    }


    return (<>
<div>

    <div className="modal fade" id="registerModal">
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content">


          <div className="modal-header">
            <h4 className="modal-title">Register account</h4>
            <button type="button" className="btn-close" data-bs-dismiss="modal"></button>
          </div>

          <form>


              <div className="modal-body">

                <div className="mb-3 mt-3">
                    <label htmlFor="register_email" className="form-label">Email:</label>
                    <input type="email" className="form-control" id="register_email" placeholder="Enter email" name="email"/>
                </div>

                <div className="mb-3">
                    <label htmlFor="register_password" className="form-label">Password:</label>
                    <input type="password" className="form-control" id="register_password" placeholder="Enter password" name="pswd"/>
                </div>

                <div className="form-check">
                  <input type="checkbox" className="form-check-input" onclick="show_register_password()"/>
                  <label className="form-check-label">Show Password</label>
                </div> 


             </div>

             <div className="modal-footer">
                <button id="submit_register_btn" type="button" className="btn btn-primary">Register</button>
             </div>
          </form>

        </div>
      </div>
    </div>



    <div className="modal fade" id="myModal">
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content">


          <div className="modal-header">
            <h4 className="modal-title">Login to account</h4>
            <button type="button" className="btn-close" data-bs-dismiss="modal"></button>
          </div>

          <form>


              <div className="modal-body">

                <div className="mb-3 mt-3">
                    <label htmlFor="email" className="form-label">Email:</label>
                    <input type="email" className="form-control" id="email" placeholder="Enter email" name="email"/>
                </div>

                <div className="mb-3">
                    <label htmlFor="pwd" className="form-label">Password:</label>
                    <input type="password" className="form-control" id="pass" placeholder="Enter password" name="pswd"/>
                </div>

                <div className="form-check">
                  <input type="checkbox" className="form-check-input" onclick="show_login_password()"/>
                  <label className="form-check-label">Show Password</label>
                </div> 


             </div>

             <div className="modal-footer">
                <button id="submit_login_btn" type="button" className="btn btn-primary">Login</button>
             </div>
          </form>

        </div>
      </div>
    </div>




    <div id="fun" className="container-fluid p-3 bg-primary text-white text-center">
      <h1>Creative Bots</h1>
      <p>Create and talk to chatbots!</p> 
    </div>

    <div className="container"> 


        <div className="offcanvas offcanvas-start text-bg-dark" id="demo">
          <div className="offcanvas-header">
            <h1 className="offcanvas-title">Settings</h1>
            <button type="button" className="btn-close btn-close-white text-reset" data-bs-dismiss="offcanvas"></button>
          </div>
          <div className="offcanvas-body">



            <button id="register_btn" type="button" className="btn btn-secondary" data-bs-toggle="modal" data-bs-target="#registerModal">Register</button>
            <br/>
            <br/>
            <button id="login_btn" type="button" className="btn btn-primary" data-bs-toggle="modal" data-bs-target="#myModal">Login</button>
            <button id="logout_btn" type="button" className="btn btn-danger text-white">Logout</button>

            <br/>
            <br/>

            <label htmlFor="system_prompt">System prompt:</label>
            <textarea id="system_prompt" className="form-control" rows="8" name="text"></textarea> 

            <br/>

            <button onClick={add_geo_location} onclick="add_geo_location()" type="button" className="btn btn-secondary text-white">Add Geo-Location</button>



            <br/>
            <br/>

            <label htmlFor="views">Choose a view:</label>
            <select name="views" id="view_select" className="form-select">
              <option value="md">Markdown</option>
              <option value="plain">Plain text</option>
            </select> 

          </div>
        </div>

        <div style={{height: '5px !important'}}></div>


        <ul className="nav nav-tabs">
          <li className="nav-item">
            <a className="nav-link active" data-bs-toggle="tab" href="#home">Chat</a>
          </li>
          <li className="nav-item">
            <a id="tab2" className="nav-link disabled" data-bs-toggle="tab" href="#create_bot_tab">Create bot</a>
          </li>

          <li className="nav-item">
            <a id="tab3" className="nav-link disabled" data-bs-toggle="tab" href="#tweak_bot_tab">Tweak bot</a>
          </li>

          <li className="nav-item">
            <a id="tab4" className="nav-link disabled" data-bs-toggle="tab" href="#question_templates_tab">Question templates</a>
          </li>

        </ul>


        <div className="tab-content">
            <div className="tab-pane container active" id="home">

                <div style={{height: '10px !important'}}></div>

                <div id="scroll_div" className="card container" style={{overflow: 'scroll', height: '400px'}}> 
                    <table id="log" className="table" style={{width: '100%'}}></table>
                </div>

                <br/>
                <div className="input-group">
                    <span className="input-group-text">
                        <select name="bots" id="bot_select" className="form-select"></select> 
                    </span>
                    <input className="form-control" list="questions" name="question" id="user_input" placeholder="What is..."/>
                    <datalist id="questions">
                      <option value="Write all the ministries of Germany and their suborganizations in dot lang and return the source code!" />
                      <option value="What is a whale?" />
                      <option value="Is a monad a burito?" />
                      <option value="Give the JSON of a graph linking Germanys 9 biggest cities" />
                    </datalist> 
                    <button id="submit_btn" className="btn btn-success" type="button">❯</button>
                </div>
                <div className="input-group">

                  <button className="btn btn-light" type="button" data-bs-toggle="offcanvas" data-bs-target="#demo">
                  Settings...
                  </button> 
                </div>


            </div>
            <div className="tab-pane container fade" id="create_bot_tab">

                <div style={{height: '10px !important'}}></div>

                <form>


                <label htmlFor="bot_name" className="form-label">Name:</label>
                <input type="bot_name" className="form-control" id="bot_name" placeholder="The displayed name of the bot."/>

                <br/>

                <label htmlFor="bot_visibility">Visibility:</label>
                <select name="bot_visibility" id="bot_visibility_select" className="form-select">
                  <option value="public">Public to All</option>
                  <option value="private">Private to User</option>
                </select> 


                <br/>

                <label htmlFor="bot_description">Description:</label>
                <textarea id="bot_description" className="form-control" rows="8" name="text" placeholder="A description of the bot and it's purpose."></textarea>

                <br/>

                <label htmlFor="bot_llm">Language model:</label>
                <select name="bot_llm" id="bot_llm_select" className="form-select">
                  <option value="llama3">Llama3</option>
                </select> 

                <br/>

                <label htmlFor="bot_system_prompt">System prompt:</label>
                <textarea id="bot_system_prompt" className="form-control" rows="8" name="text" placeholder="The prompt that defines the bot's main behaviour."></textarea>

                <hr/>

                <div className="row">
                      <div className="col"></div>
                      <div className="col"></div>
                      <div className="col"></div>
                      <div className="col"></div>
                      <div className="col">
                        <button id="create_bot_btn" disabled type="button" className="btn btn-primary text-white">Create bot</button>
                     </div>
                </div>

                <br/>



                <div id="alert_spawn"></div>

                </form>

            </div>


            <div className="tab-pane container fade" id="tweak_bot_tab">

                <div style={{height: '10px !important'}}></div>
                              

                <form>


                <label htmlFor="change_bots">Choose a bot:</label>
                <select name="change_bots" id="change_bot_select" className="form-select">
                    <option value="md">Some bot</option>
                </select>

                <br/>

                <label htmlFor="bot_name" className="form-label">Name:</label>
                <input type="bot_name" className="form-control" id="change_bot_name" placeholder="The displayed name of the bot."/>

                <br/>

                <label htmlFor="bot_visibility">Visibility:</label>
                <select name="bot_visibility" id="change_bot_visibility_select" className="form-select">
                  <option value="public">Public to All</option>
                  <option value="private">Private to User</option>
                </select> 

                <br/>

                <label htmlFor="bot_description">Description:</label>
                <textarea id="change_bot_description" className="form-control" rows="8" name="text" placeholder="A description of the bot and it's purpose."></textarea>

                <br/>

                <label htmlFor="bot_llm">Language model:</label>
                <select name="bot_llm" id="change_bot_llm_select" className="form-select">
                  <option value="llama3">Llama3</option>
                </select> 

                <br/>

                <label htmlFor="bot_system_prompt">System prompt(behavior):</label>
                <textarea id="change_bot_system_prompt" className="form-control" rows="8" name="text" placeholder="The prompt that defines the bot's main behaviour."></textarea>

                <br/>

                <h4>Knowledge resources:</h4>
                <br/>

                <label htmlFor="change_bot_rag_text_name">Text:</label>
                <textarea id="change_bot_rag_text" className="form-control" rows="16" name="change_bot_rag_text_name" placeholder="A text that contains information used for the bot's answers.'"></textarea>

                <br/>

                <label htmlFor="avatar">Text documents:</label>
                <input disabled className="form-control" type="file" id="avatar" name="avatar" multiple accept=".pdf,.xml,.txt,.md,.doc,.docx,.odt,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document" />

                <br/>

                <label htmlFor="rag_urls">Links:</label>
                <div className="input-group">
                    <input disabled className="form-control" type="url" name="rag_urls" id="url" placeholder="https://example.com" pattern="https://.*" size="30" />
                    <button disabled id="add_url_btn" className="btn btn-success" type="button">Add</button>
                </div>




                <hr/>

                <div className="row">
                      <div className="col"></div>
                      <div className="col"></div>
                      <div className="col"></div>
                      <div className="col">
                        <button id="change_bot_btn" disabled type="button" className="btn btn-primary text-white">Change bot</button>
                      </div>
                      <div className="col">
                        <button id="delete_bot_btn" disablxed type="button" className="btn btn-danger text-white">Delete bot</button>
                     </div>
                </div>

                <br/>



                <div id="change_bot_alert_spawn"></div>


                </form>

            </div>
        </div>


    </div>
</div>



</>
  );
}
